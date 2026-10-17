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
            kvGet('ind_snap_hist'),
        ]).then(function(results) {
            var kvWlData    = results[0];
            var kvWlOrder   = results[1];
            var kvWlLast    = results[2];
            var kvPresets   = results[3];
            var kvIndHist   = results[4];

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

            // Industry snapshot history — drives the rank deltas AND the breadth (UPTR% / 50MA% / 21EMA%) arrows.
            // Keyed by TRADING DAY (derived from the data's own built_at), not by when this browser happened to
            // load. One entry per trading day (the latest build of that day wins), so extra rebuilds/reloads on the
            // same day, or over a weekend, can no longer overwrite the previous day's baseline. The baseline is the
            // most recent stored day BEFORE the day the current data is for.
            var LS_IND_HIST   = 'ind_snap_hist';
            var IND_HIST_KEEP = 10;

            // Prev-snapshot loader: sets both the rank deltas and the breadth arrows
            function _setPrevSnap(s) { indPrevRanks = s.ranks; indPrevBreadth = s.breadth || {}; }

            function _parseHist(str) {
                try { var h = str ? JSON.parse(str) : null; return (h && typeof h === 'object') ? h : null; } catch(e) { return null; }
            }

            // Union of the KV and localStorage copies; for the same day keep the later build.
            function _mergeHist(a, b) {
                var out = {};
                [a, b].forEach(function(h) {
                    if (!h) return;
                    Object.keys(h).forEach(function(day) {
                        var en = h[day];
                        if (!en || !en.ranks) return;
                        if (!out[day] || String(en.built_at || '') > String(out[day].built_at || '')) out[day] = en;
                    });
                });
                return out;
            }

            // Trading day a build represents, in America/New_York. A weekday build at/after 16:00 ET is that day's
            // close; a build before the open, or on a weekend, is still the previous weekday's close. A build made
            // while the market is open (09:30-16:00 ET, partial bar) is for the current day and is never recorded.
            // Exchange holidays are not modelled: a rebuild on a holiday counts as a new day with unchanged data.
            function _tradingDayOf(builtAt) {
                var dt = builtAt ? new Date(builtAt) : null;
                if (!dt || isNaN(dt.getTime())) return null;
                var p = {};
                try {
                    new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', hourCycle: 'h23' })
                        .formatToParts(dt).forEach(function(x) { p[x.type] = x.value; });
                } catch(e) { return null; }
                var d = new Date(Date.UTC(+p.year, +p.month - 1, +p.day));
                var mins = ((+p.hour) % 24) * 60 + (+p.minute);
                function isWk(x) { var w = x.getUTCDay(); return w !== 0 && w !== 6; }
                var mid = isWk(d) && mins >= 570 && mins < 960;
                if (!mid && !(isWk(d) && mins >= 960)) {
                    do { d.setUTCDate(d.getUTCDate() - 1); } while (!isWk(d));
                }
                var day = d.getUTCFullYear() + '-' + String(d.getUTCMonth() + 1).padStart(2, '0') + '-' + String(d.getUTCDate()).padStart(2, '0');
                return { day: day, mid: mid };
            }

            var _hist    = _mergeHist(_parseHist(kvIndHist), _parseHist(localStorage.getItem(LS_IND_HIST)));
            var _builtAt = (snapshot && snapshot.built_at) || (industriesData && industriesData.built_at) || null;
            var _td      = _tradingDayOf(_builtAt);

            if (_td) {
                // Baseline = latest stored trading day strictly before the day the current data is for
                var _days = Object.keys(_hist).sort();
                for (var _i = _days.length - 1; _i >= 0; _i--) {
                    if (_days[_i] < _td.day) { _setPrevSnap(_hist[_days[_i]]); break; }
                }

                // Record today's data under its trading day (latest build of the day wins; skip mid-session builds)
                if (!_td.mid && industriesData && industriesData.industries && snapshot && snapshot.industry_summary) {
                    var newRanks = {};
                    industriesData.industries.forEach(function(ind) {
                        if (ind.rank != null) newRanks[ind.industry] = ind.rank;
                    });
                    var newBreadth = {};
                    Object.keys(snapshot.industry_summary).forEach(function(k) {
                        var bs = snapshot.industry_summary[k];
                        newBreadth[k] = [bs.pct_confirmed_uptrend, bs.pct_above_50sma, bs.pct_above_21ema];
                    });
                    var _curEntry = _hist[_td.day];
                    if (Object.keys(newRanks).length && Object.keys(newBreadth).length && !(_curEntry && _curEntry.built_at === _builtAt)) {
                        _hist[_td.day] = { ranks: newRanks, breadth: newBreadth, built_at: _builtAt };
                        Object.keys(_hist).sort().slice(0, -IND_HIST_KEEP).forEach(function(dk) { delete _hist[dk]; });
                        var _histStr = JSON.stringify(_hist);
                        kvSet(LS_IND_HIST, _histStr);
                        try { localStorage.setItem(LS_IND_HIST, _histStr); } catch(e) {}
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
        // Market movers — right-click to open the watchlist quick-picker
        // (same pattern as stocks.js / scans.js / the scan-nav-panel below)
        document.getElementById('view-market').addEventListener('contextmenu', function(e) {
            var row = e.target.closest('.gl-clickable');
            if (!row) return;
            e.preventDefault();
            var ticker = row.getAttribute('data-ticker');
            if (!ticker) return;
            var fakeBtn = {
                getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
                getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                _wlNoSwitch: true
            };
            wlOpenPicker(fakeBtn, e, false);
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
                _mcFsChart.timeScale().setVisibleLogicalRange({ from: n - _mcFsVisibleBars, to: n + 15 });
            }
            return;
        }
        // WL side panel — gated on currentView too, since wl-chart-widget's
        // display isn't reset when you tab away without explicitly closing the chart.
        if (currentView === 'watchlists') {
            var wlWidget = document.getElementById('wl-chart-widget');
            if (wlWidget && wlWidget.style.display === 'block') {
                if (_wlChart && _wlOhlcv.length) {
                    var n = _wlOhlcv.length;
                    _wlChart.timeScale().setVisibleLogicalRange({ from: n - _wlVisibleBars, to: n + 15 });
                }
                return;
            }
        }
        // AL side panel — same reasoning: alChartPanelClose() only runs on an
        // explicit close, not on tab-switch, so gate on currentView as well.
        if (currentView === 'alerts') {
            var alPanel = document.getElementById('al-chart-panel');
            if (alPanel && alPanel.classList.contains('open')) {
                if (_alChart && _alOhlcv.length) {
                    var n = _alOhlcv.length;
                    _alChart.timeScale().setVisibleLogicalRange({ from: n - _alVisibleBars, to: n + 15 });
                }
                return;
            }
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

