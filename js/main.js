    // ── Load data ─────────────────────────────────────────────────────────
    Promise.all([
        fetch('data/snapshot.json').then(function(r){ return r.ok ? r.json() : null; }).catch(function(){ return null; }),
        fetch('data/industries.json').then(function(r){ return r.ok ? r.json() : null; }).catch(function(){ return null; }),
    ]).then(function(res) {
        snapshot       = res[0];
        industriesData = res[1];

        // Build ticker lookup map for O(1) access in chart modal
        tickerMap = {};
        if (snapshot && snapshot.by_industry) {
            Object.keys(snapshot.by_industry).forEach(function(ind) {
                snapshot.by_industry[ind].forEach(function(row) {
                    if (row.ticker) { tickerMap[row.ticker] = row; row._snapPrice = row.price; }
                });
            });
        }

        document.getElementById('loading-msg').style.display = 'none';

        // Kick off live intraday Day% fetch and refresh every 60s — but only
        // queue the actual batch requests while on a view that uses this data
        // (industries/sector/market). Otherwise this was firing every 60s
        // regardless of page, competing with multichart's grid loads for the
        // shared yahoo-proxy-pace.js clock even while nobody was looking at
        // the industry heatmap at all. Mirrors the currentView check market.js
        // already does for its own timer (see market.js, marketTimer).
        fetchLiveIndustryDay();
        if (!_indLiveDayInterval) {
            _indLiveDayInterval = setInterval(function() {
                if (currentView !== 'industries' && currentView !== 'sector' && currentView !== 'market') return;
                if (_mcFsIsOpen()) return; // fullscreen chart open — don't kick off a fresh burst
                fetchLiveIndustryDay();
            }, 60000);
        }

        // Sync DistMA buttons
        document.querySelectorAll('.dist-ma-type-btn').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-type') === activeMAType); });
        document.querySelectorAll('.dist-ma-len-btn').forEach(function(b){ b.classList.toggle('active', parseInt(b.getAttribute('data-len')) === activeMALength); });

        // Restore data from KV if localStorage is empty (fresh browser/device)
        Promise.all([
            kvGet('wl_data'),
            kvGet('wl_order'),
            kvGet('wl_last'),
            kvGet('scan_presets'),
            kvGet('ind_rank_snapshot'),
        ]).then(function(results) {
            var kvWlData    = results[0];
            var kvWlOrder   = results[1];
            var kvWlLast    = results[2];
            var kvPresets   = results[3];
            var kvIndSnap   = results[4];

            // Only restore from KV if localStorage doesn't already have data
            if (kvWlData && !localStorage.getItem(LS_WL_KEY)) {
                try { localStorage.setItem(LS_WL_KEY, kvWlData); } catch(e) {}
            }
            if (kvWlOrder && !localStorage.getItem(LS_WL_ORD_KEY)) {
                try { localStorage.setItem(LS_WL_ORD_KEY, kvWlOrder); } catch(e) {}
            }
            if (kvWlLast && !localStorage.getItem(LS_WL_LAST_KEY)) {
                try { localStorage.setItem(LS_WL_LAST_KEY, kvWlLast); } catch(e) {}
            }
            if (kvPresets && !localStorage.getItem(LS_PRESETS_KEY)) {
                try { localStorage.setItem(LS_PRESETS_KEY, kvPresets); } catch(e) {}
            }

            // Industry rank snapshot — load previous ranks for delta display,
            // then save today's ranks only when ranks actually changed (so weekend
            // deltas persist until Monday's data arrives).
            // Fix 1: use local date, not UTC, to avoid midnight boundary issues.
            // Fix 2: mirror snapshots to localStorage as fallback for KV failures.
            var _dn = new Date();
            var today = _dn.getFullYear() + '-' + String(_dn.getMonth()+1).padStart(2,'0') + '-' + String(_dn.getDate()).padStart(2,'0');
            var LS_IND_SNAP      = 'ind_rank_snapshot';
            var LS_IND_SNAP_PREV = 'ind_rank_snapshot_prev';

            function _parseSnap(str) { try { return str ? JSON.parse(str) : null; } catch(e) { return null; } }

            // Load current snapshot — KV first, localStorage fallback
            var prevSnap = _parseSnap(kvIndSnap) || _parseSnap(localStorage.getItem(LS_IND_SNAP));

            if (prevSnap && prevSnap.ranks) {
                if (prevSnap.date !== today) {
                    // Snapshot is from a previous day.
                    // If ranks are identical to current data, we're in a weekend/holiday period —
                    // use the archived _prev snapshot (e.g. Thursday's ranks) so Friday's delta
                    // stays visible through Saturday, Sunday, and until Monday's data changes.
                    var _curRanksCheck = {};
                    if (industriesData && industriesData.industries) {
                        industriesData.industries.forEach(function(ind) {
                            if (ind.rank != null) _curRanksCheck[ind.industry] = ind.rank;
                        });
                    }
                    var _ranksUnchanged = Object.keys(_curRanksCheck).length > 0 &&
                        JSON.stringify(_curRanksCheck) === JSON.stringify(prevSnap.ranks);

                    if (_ranksUnchanged) {
                        // Weekend/holiday: load _prev (e.g. Thursday) so delta reflects Friday's move
                        var lsPrevWknd = _parseSnap(localStorage.getItem(LS_IND_SNAP_PREV));
                        if (lsPrevWknd && lsPrevWknd.ranks) {
                            indPrevRanks = lsPrevWknd.ranks;
                        } else {
                            kvGet('ind_rank_snapshot_prev').then(function(kvIndSnapPrev) {
                                var ySnap = _parseSnap(kvIndSnapPrev);
                                if (ySnap && ySnap.ranks) {
                                    indPrevRanks = ySnap.ranks;
                                    try { localStorage.setItem(LS_IND_SNAP_PREV, JSON.stringify(ySnap)); } catch(e) {}
                                    renderIndustries();
                                }
                            });
                        }
                    } else {
                        // Normal weekday — snapshot is genuinely yesterday's, use it directly
                        indPrevRanks = prevSnap.ranks;
                    }
                } else {
                    // Snapshot is already today's — load the archived yesterday snapshot.
                    // Try localStorage first (instant, reliable for same-browser reloads).
                    var lsPrev = _parseSnap(localStorage.getItem(LS_IND_SNAP_PREV));
                    if (lsPrev && lsPrev.ranks) {
                        indPrevRanks = lsPrev.ranks;
                    } else {
                        kvGet('ind_rank_snapshot_prev').then(function(kvIndSnapPrev) {
                            var yesterdaySnap = _parseSnap(kvIndSnapPrev);
                            if (yesterdaySnap && yesterdaySnap.ranks) {
                                indPrevRanks = yesterdaySnap.ranks;
                                try { localStorage.setItem(LS_IND_SNAP_PREV, JSON.stringify(yesterdaySnap)); } catch(e) {}
                                renderIndustries();
                            }
                        });
                    }
                }
            }

            if (industriesData && industriesData.industries) {
                // Build the new rank map from current data
                var newRanks = {};
                industriesData.industries.forEach(function(ind) {
                    if (ind.rank != null) newRanks[ind.industry] = ind.rank;
                });

                // Fix 3: only archive + overwrite when ranks actually changed.
                // This keeps Friday's snapshot as _prev through the whole weekend
                // so deltas remain visible until Monday's data brings real changes.
                var currentRanks = prevSnap ? prevSnap.ranks : null;
                var ranksChanged = !currentRanks || JSON.stringify(newRanks) !== JSON.stringify(currentRanks);

                if (ranksChanged) {
                    // Archive old snapshot as "yesterday" before overwriting
                    if (prevSnap && prevSnap.ranks) {
                        var prevStr = JSON.stringify(prevSnap);
                        kvSet('ind_rank_snapshot_prev', prevStr);
                        try { localStorage.setItem(LS_IND_SNAP_PREV, prevStr); } catch(e) {}
                    }
                    var newSnap = { date: today, ranks: newRanks };
                    var newSnapStr = JSON.stringify(newSnap);
                    kvSet('ind_rank_snapshot', newSnapStr);
                    try { localStorage.setItem(LS_IND_SNAP, newSnapStr); } catch(e) {}
                } else if (!prevSnap || prevSnap.date !== today) {
                    // Ranks unchanged but it's a new calendar day — just update the date
                    // so same-day logic works, without disturbing _prev.
                    // Skip on weekends (Sat=6, Sun=0): bumping the date on Sat would make
                    // Sunday treat Saturday's snap as "yesterday", wiping the Friday delta.
                    var _dow = new Date().getDay();
                    if (_dow !== 0 && _dow !== 6) {
                        var updatedSnap = { date: today, ranks: newRanks };
                        var updatedSnapStr = JSON.stringify(updatedSnap);
                        kvSet('ind_rank_snapshot', updatedSnapStr);
                        try { localStorage.setItem(LS_IND_SNAP, updatedSnapStr); } catch(e) {}
                    }
                }
            }

            showView('market');
            sfRenderPills();
            wlAttachListeners();
            alLoad();
        }).catch(function() {
            // KV unavailable — proceed normally with localStorage
            showView('market');
            sfRenderPills();
            wlAttachListeners();
            alLoad();
        });

        // Market movers — click to open chart modal
        document.getElementById('view-market').addEventListener('click', function(e) {
            var row = e.target.closest('.gl-clickable');
            if (!row) return;
            var ticker   = row.getAttribute('data-ticker');
            var industry = row.getAttribute('data-industry') || '';
            if (ticker) openChartModal(ticker);
        });
        document.getElementById('scans-table-view').addEventListener('dblclick', function(e) {
            var row = e.target.closest('.stock-row');
            if (!row) return;
            openChartModal(row.getAttribute('data-symbol'));
        });
    });
    // ── SCAN NAV PANEL (removable block) ─────────────────────────────────
    var _snpData   = [];
    var _snpIndex  = -1;
    var _snpSource = '';

    function snpBuild() {
        var panel = document.getElementById('scan-nav-panel');
        if (!panel) return;

        var data = [];
        _snpSource = currentView;

        // Set title and font-size based on view
        var titleEl = document.getElementById('snp-title');
        if (titleEl) {
            if (currentView === 'scans') {
                titleEl.textContent = 'Results';
                titleEl.style.fontSize = '13px';
            } else if (currentView === 'industry-stocks') {
                titleEl.textContent = 'Group';
                titleEl.style.fontSize = '13px';
            } else {
                titleEl.textContent = 'List';
                titleEl.style.fontSize = '10px';
            }
        }

        if (currentView === 'scans') {
            if (!sfRows.length) { snpHide(); return; }
            data = _vsData.map(function(r) {
                return { ticker: r.ticker, rs3m: r.weighted_rs_pct };
            });
        } else if (currentView === 'industry-stocks') {
            var rows = Array.from(document.querySelectorAll('#stocks-tbody .stock-row'));
            if (!rows.length) { snpHide(); return; }
            data = rows.map(function(r) {
                var sym = r.getAttribute('data-symbol');
                var sd  = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                return { ticker: sym, rs3m: sd ? sd.weighted_rs_pct : null };
            });
        } else {
            snpHide(); return;
        }

        if (!data.length) { snpHide(); return; }

        // Sort by 3M RS descending
        data.sort(function(a, b) {
            var av = a.rs3m != null ? a.rs3m : -1;
            var bv = b.rs3m != null ? b.rs3m : -1;
            return bv - av;
        });

        _snpData = data;

        var countEl = document.getElementById('snp-count');
        if (countEl) countEl.textContent = data.length;

        var html = '';
        data.forEach(function(d, i) {
            html += '<div class="snp-row" data-snp-idx="' + i + '">' +
                '<span class="snp-rank">' + (i + 1) + '</span>' +
                '<span class="snp-ticker">' + d.ticker + '</span>' +
            '</div>';
        });

        var listEl = document.getElementById('snp-list');
        if (listEl) {
            listEl.innerHTML = html;
            if (typeof alStampBadges === 'function') alStampBadges();
            tickerHoverBind(listEl, '.snp-ticker');
            listEl.onclick = function(e) {
                var row = e.target.closest('.snp-row');
                if (!row) return;
                var idx = parseInt(row.getAttribute('data-snp-idx'), 10);
                if (e.target.closest('.al-ticker-pill')) {
                    var d = _snpData[idx];
                    if (d && typeof alGoToTicker === 'function') alGoToTicker(d.ticker);
                    return;
                }
                snpNavigateTo(idx);
            };
            listEl.oncontextmenu = function(e) {
                var row = e.target.closest('.snp-row');
                if (!row) return;
                e.preventDefault();
                var idx = parseInt(row.getAttribute('data-snp-idx'), 10);
                var d = _snpData[idx];
                if (!d) return;
                var fakeBtn = {
                    getAttribute: function(attr) { return attr === 'data-ticker' ? d.ticker : null; },
                    getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                    _wlNoSwitch: true
                };
                wlOpenPicker(fakeBtn, e, false);
            };
        }

        panel.classList.add('snp-open');
    }

    function snpHide() {
        var panel = document.getElementById('scan-nav-panel');
        if (panel) panel.classList.remove('snp-open');
        _snpData  = [];
        _snpIndex = -1;
    }

    function snpSetActive(idx) {
        _snpIndex = idx;
        var listEl = document.getElementById('snp-list');
        if (!listEl) return;
        listEl.querySelectorAll('.snp-row').forEach(function(r) {
            r.classList.toggle('snp-active', parseInt(r.getAttribute('data-snp-idx'), 10) === idx);
        });
        var activeRow = listEl.querySelector('.snp-row.snp-active');
        if (activeRow) activeRow.scrollIntoView({ block: 'nearest' });
    }

    function snpNavigateTo(idx) {
        if (!_snpData.length) return;
        idx = Math.max(0, Math.min(idx, _snpData.length - 1));
        snpSetActive(idx);
        var d = _snpData[idx];
        if (!d) return;

        if (_snpSource === 'scans') {
            currentStockIndex = idx;
            allStockRows = { length: _vsData.length };
        } else if (_snpSource === 'industry-stocks') {
            currentStockIndex = idx;
            var rows = Array.from(document.querySelectorAll('#stocks-tbody .stock-row'));
            rows.forEach(function(r, i) { r.classList.toggle('active', i === idx); });
            if (rows[idx]) rows[idx].scrollIntoView({ block: 'nearest' });
        }

        openChartModal(d.ticker);
    }

    // Wrap openChartModal to use LW fullscreen chart + side panel (replaces TradingView modal)
    openChartModal = window.openChartModal = function(symbol) {
        openMcFullscreen(symbol, _mcFsTf || 'D');
        if (!_snpData.length) {
            snpBuild();
        }
        var idx = _snpData.findIndex(function(d) { return d.ticker === symbol; });
        if (idx !== -1) snpSetActive(idx);
    };

    // Wrap closeChartModal to close LW fullscreen + hide panel
    var _alGoToTickerClosing = false;
    window.closeChartModal = function() {
        closeMcFullscreen();
        snpHide();
        if (!_alGoToTickerClosing) { _alReturnState = null; _scanReturnState = null; }
        // Restore industry stocks table scroll position after modal closes
        if (currentView === 'industry-stocks' && _indStocksScrollBeforeModal > 0) {
            var _savedScroll = _indStocksScrollBeforeModal;
            setTimeout(function() {
                var _stWrap = document.querySelector('#view-industry-stocks .stocks-table-wrap');
                if (_stWrap) _stWrap.scrollTop = _savedScroll;
            }, 0);
        }
    };

    // Arrow key nav when LW fullscreen is open — capture phase so it takes priority
    document.addEventListener('keydown', function(e) {
        if (!_snpData.length) return;
        var modalOpen = document.getElementById('mc-fullscreen-overlay').classList.contains('open');
        if (!modalOpen) return;
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'ArrowDown') { e.preventDefault(); e.stopPropagation(); snpNavigateTo(_snpIndex + 1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); e.stopPropagation(); snpNavigateTo(_snpIndex - 1); }
    }, true);

    // Alt+R — reset active LW chart to its original opening zoom/position
    document.addEventListener('keydown', function(e) {
        if (!e.altKey || e.key !== 'r') return;
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        e.preventDefault();
        // Fullscreen takes priority
        if (document.getElementById('mc-fullscreen-overlay').classList.contains('open')) {
            if (_mcFsChart && _mcFsOhlcv.length) {
                var n = _mcFsOhlcv.length;
                _mcFsChart.timeScale().setVisibleLogicalRange({ from: n - _mcFsVisibleBars, to: n + 12 });
            }
            return;
        }
        // WL side panel
        if (document.getElementById('wl-chart-panel').classList.contains('open')) {
            if (_wlChart && _wlOhlcv.length) {
                var n = _wlOhlcv.length;
                _wlChart.timeScale().setVisibleLogicalRange({ from: n - _wlVisibleBars, to: n + 12 });
            }
            return;
        }
        // AL side panel
        if (document.getElementById('al-chart-panel').classList.contains('open')) {
            if (_alChart && _alOhlcv.length) {
                var n = _alOhlcv.length;
                _alChart.timeScale().setVisibleLogicalRange({ from: n - _alVisibleBars, to: n + 12 });
            }
            return;
        }
    });
    // ── END SCAN NAV PANEL ───────────────────────────────────────────────

    // ── Floating tooltip for data-rs-tip elements ────────────────────────
    (function() {
        var tip = document.createElement('div');
        tip.id = 'rs-float-tip';
        document.body.appendChild(tip);
        var _tipTarget = null;
        document.addEventListener('mouseover', function(e) {
            var el = e.target.closest('[data-rs-tip]');
            if (el === _tipTarget) return;
            _tipTarget = el;
            if (!el) {
                tip.style.display = 'none';
            } else {
                tip.textContent = el.getAttribute('data-rs-tip');
                tip.style.display = 'block';
            }
        });
        document.addEventListener('mousemove', function(e) {
            if (!_tipTarget) return;
            tip.style.left = (e.clientX + 12) + 'px';
            tip.style.top  = (e.clientY - 32) + 'px';
        });
    })();

// ── Cross-module ticker data accessor (used by market-popup.js) ──────────
    window._getTickerPopupData = function(ticker) {
        var row = tickerMap ? tickerMap[ticker] : null;
        var name = (row && row.name) ? row.name : '';
        var price = null, prevClose = null;
        // Only trust the live caches (scans/watchlist/alerts) while the market's
        // open. Each one is populated unconditionally by its owning file — outside
        // market hours "live" price equals the snapshot close, so trusting it here
        // would show a spurious 0 instead of falling through to the real last-
        // session change below.
        if (wlIsMarketOpen()) {
            var slp = scanLivePrices[ticker];
            if (slp && slp.price) { price = slp.price; prevClose = slp.prevClose || null; }
            if (!price) {
                var wlp = wlLivePrices[ticker];
                if (wlp && wlp.price) { price = wlp.price; prevClose = wlp.prevClose || null; }
            }
            if (!price && alertPrices[ticker]) {
                price = alertPrices[ticker];
                prevClose = alertPrevClose[ticker] || null;
            }
            // If we resolved a price from scan/wl but prevClose is still null,
            // fall back to alertPrevClose so the popup chg matches the alerts table.
            if (price && !prevClose && alertPrevClose[ticker]) {
                prevClose = alertPrevClose[ticker];
            }
        }
        if (!price && row && row.price != null) { price = row.price; }
        if (!prevClose && row && row.price != null && row.daily != null) {
            prevClose = row.price / (1 + row.daily / 100);
        }
        var rs   = (row && row.Percentile      != null) ? row.Percentile      : null;
        var rs3m = (row && row.weighted_rs_pct != null) ? row.weighted_rs_pct : null;
        var indRank = null, indTotal = null, indPct = null;
        var industry = (row && row.industry) ? row.industry : null;
        if (industry && typeof industriesData !== 'undefined' && industriesData && industriesData.industries) {
            var indRow = industriesData.industries.find(function(x) { return x.industry === industry; });
            indTotal = industriesData.industries.length;
            if (indRow && indRow.rank != null) {
                indRank = indRow.rank;
                indPct  = indRow.percentile != null ? indRow.percentile : null;
            }
        }
        return { name: name, price: price, prevClose: prevClose, rs: rs, rs3m: rs3m, indRank: indRank, indTotal: indTotal, indPct: indPct, industry: industry };
    };

