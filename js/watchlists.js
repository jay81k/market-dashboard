    // ── Watchlist live prices ─────────────────────────────────────────────
    function wlIsMarketOpen() {
        var now = new Date();
        var et  = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
        var day = et.getDay();
        if (day === 0 || day === 6) return false;
        var h = et.getHours(), m = et.getMinutes();
        var mins = h * 60 + m;
        return mins >= 570 && mins < 960; // 9:30–16:00 ET
    }

    function wlFetchOneTicker(ticker) {
        var url = WL_PROXY + '?symbol=' + encodeURIComponent(ticker) + '&interval=1d&range=2d';
        return fetch(url).then(function(r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function(data) {
            var result = data && data.chart && data.chart.result && data.chart.result[0];
            if (!result) return null;
            var price     = result.meta && result.meta.regularMarketPrice;
            var prevClose = result.meta && result.meta.previousClose;
            // If meta.previousClose is missing, derive it from the first bar in the chart data
            if ((!prevClose || prevClose <= 0) && result.indicators) {
                var closes = result.indicators.quote &&
                             result.indicators.quote[0] &&
                             result.indicators.quote[0].close;
                if (closes && closes.length >= 2) {
                    prevClose = closes[0];
                }
            }
            if (!price || price <= 0) return null;
            return { price: price, prevClose: prevClose || null };
        }).catch(function() { return null; });
    }

    function wlFetchPrices() {
        var active  = wlGetLastList();
        var all     = wlGetAll();
        var tickers = active && all[active] ? all[active] : [];
        if (!tickers.length) return;

        var url = WL_PROXY + '?action=quotes_batch&tickers=' + tickers.map(encodeURIComponent).join(',');
        fetch(url).then(function(r) {
            return r.ok ? r.json() : null;
        }).then(function(data) {
            if (!data || !data.quotes) return;
            data.quotes.forEach(function(q) {
                if (q && q.ticker && q.price) {
                    wlLivePrices[q.ticker] = {
                        price:     q.price,
                        prevClose: q.prevClose || null,
                        updatedAt: new Date()
                    };
                }
            });
            wlUpdatePriceRows();
        }).catch(function() {});
    }

    function wlUpdatePriceRows() {
        if (_wlSortCol === 'chgp') { wlRender(); return; }
        document.querySelectorAll('.wl-ticker-row').forEach(function(row) {
            var t    = row.getAttribute('data-wl-ticker');
            var live = wlLivePrices[t];
            if (!live || !live.price) return;
            var price     = live.price;
            var prevClose = live.prevClose;
            var chgAbs    = (prevClose && prevClose > 0) ? price - prevClose : null;
            var chgPct    = (prevClose && prevClose > 0) ? ((price - prevClose) / prevClose) * 100 : null;
            // Fall back to snapshot daily if no prevClose from live
            if (chgPct == null) {
                var sd = wlLookupStock(t);
                if (sd && sd.daily != null) {
                    chgPct = sd.daily;
                    chgAbs = sd.price ? (sd.price / (1 + sd.daily / 100)) * (sd.daily / 100) : null;
                }
            }
            var cl     = chgPct == null ? 'neutral' : chgPct > 0 ? 'up' : chgPct < 0 ? 'down' : 'neutral';
            var lastEl = row.querySelector('.wl-c-last');
            var chgEl  = row.querySelector('.wl-c-chg');
            var chgpEl = row.querySelector('.wl-c-chgp');
            if (lastEl) lastEl.textContent = price.toFixed(2);
            if (chgEl)  { chgEl.textContent = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '—'; chgEl.className = 'wl-c-chg ' + cl; }
            if (chgpEl) { chgpEl.textContent = chgPct != null ? (chgPct >= 0 ? '+' : '') + chgPct.toFixed(2) + '%' : '—'; chgpEl.className = 'wl-c-chgp ' + cl; }
        });
    }

    function wlStartPricePolling() {
        if (wlPriceTimer) clearInterval(wlPriceTimer);
        wlFetchPrices();
        if (!wlIsMarketOpen()) return;
        wlPriceTimer = setInterval(function() {
            if (currentView !== 'watchlists') return;
            if (!wlIsMarketOpen()) { wlStopPricePolling(); return; }
            wlFetchPrices();
        }, 60 * 1000);
    }

    function wlStopPricePolling() {
        if (wlPriceTimer) { clearInterval(wlPriceTimer); wlPriceTimer = null; }
    }

    // ── Watchlist Multichart ──────────────────────────────────────────────
    window.toggleWlMultichart = function() {
        wlMcActive = !wlMcActive;
        var btn = document.getElementById('wl-multichart-toggle-btn');
        btn.style.background  = wlMcActive ? '#1f3a5c' : '';
        btn.style.borderColor = wlMcActive ? '#388bfd' : '';
        btn.style.color       = wlMcActive ? '#58a6ff' : '';
        var settingsBar = document.getElementById('wl-chart-settings');
        document.getElementById('wl-chart-empty').style.display  = wlMcActive ? 'none' : (wlChartTicker ? 'none' : '');
        document.getElementById('wl-chart-widget').style.display = wlMcActive ? 'none' : (wlChartTicker ? 'block' : 'none');
        if (settingsBar) settingsBar.style.display = (!wlMcActive && wlChartTicker) ? 'flex' : 'none';
        document.getElementById('wl-multichart-view').style.display = wlMcActive ? 'flex' : 'none';
        document.querySelector('.wl-chart-panel-header').style.display = wlMcActive ? 'none' : '';
        if (wlMcActive) {
            _destroyWlChart();
            renderWlMc();
        } else if (wlChartTicker) {
            // Re-render the LW chart when coming back from multichart mode
            wlSelectTicker(wlChartTicker);
        }
    };

    window.setWlMcTf = function(tf) {
        wlMcTimeframe = tf;
        document.querySelectorAll('#wl-multichart-toolbar .mc-tf-btn').forEach(function(b){
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        renderWlMc();
    };

    window.setWlMcCols = function(n) {
        wlMcCols = n;
        document.querySelectorAll('#wl-multichart-toolbar .mc-col-btn').forEach(function(b){
            b.classList.toggle('active', +b.getAttribute('data-cols') === n);
        });
        document.getElementById('wl-multichart-grid').style.gridTemplateColumns = 'repeat(' + n + ', 1fr)';
    };

    function renderWlMc() {
        var grid = document.getElementById('wl-multichart-grid');
        var active  = wlGetLastList();
        var all     = wlGetAll();
        var tickers = (active && all[active]) ? all[active].slice() : [];
        if (!tickers.length) {
            grid.innerHTML = '<div style="color:#484f58;padding:20px;">No tickers in this watchlist.</div>';
            return;
        }
        // Sort by chg% highest → lowest (mirrors watchlist default)
        var tickerData = {};
        tickers.forEach(function(t) {
            var sd = wlLookupStock(t), live = wlLivePrices[t];
            tickerData[t] = (live && live.price && live.prevClose) ? ((live.price - live.prevClose) / live.prevClose) * 100 : (sd ? sd.daily : null);
        });
        tickers.sort(function(a, b) {
            var av = tickerData[a] != null ? tickerData[a] : -Infinity;
            var bv = tickerData[b] != null ? tickerData[b] : -Infinity;
            return bv - av;
        });
        _buildLwMcGrid(grid, tickers, wlMcTimeframe, wlMcCols, wlMcWidgets, 'wl');
    }

    // ── Watchlists ────────────────────────────────────────────────────────
    var WL_FLAGGED = 'Flagged';

    function wlIsFlagged(ticker) {
        var all = wlGetAll();
        return !!(all[WL_FLAGGED] && all[WL_FLAGGED].indexOf(ticker) !== -1);
    }

    window.wlFlagTicker = function(ticker, btn) {
        var all = wlGetAll();
        if (!all[WL_FLAGGED]) {
            all[WL_FLAGGED] = [];
            var order = [];
            try { order = JSON.parse(localStorage.getItem(LS_WL_ORD_KEY) || '[]'); } catch(e) {}
            if (order.indexOf(WL_FLAGGED) === -1) order.unshift(WL_FLAGGED); // put Flagged first
            wlSaveOrder(order);
            wlSetLastList(WL_FLAGGED);
        }
        var idx = all[WL_FLAGGED].indexOf(ticker);
        if (idx !== -1) {
            all[WL_FLAGGED].splice(idx, 1);
            if (btn) { btn.classList.remove('flagged'); btn.title = 'Add to Flagged'; }
        } else {
            all[WL_FLAGGED].push(ticker);
            if (btn) { btn.classList.add('flagged'); btn.title = 'Remove from Flagged'; }
        }
        wlSaveAll(all);
        if (currentView === 'watchlists' && !wlMcActive) wlRender();
        wlRefreshStars();
    };
    var LS_WL_KEY      = 'dashboard-watchlists';
    var LS_WL_ORD_KEY  = 'dashboard-watchlists-order';
    var LS_WL_LAST_KEY = 'dashboard-watchlists-last';
    var wlChartTf      = 'D';
    var wlChartTicker  = null;
    var wlChartWidget  = null;
    var wlMcActive     = false;
    var wlMcTimeframe  = 'D';
    var wlMcCols       = parseInt(localStorage.getItem('mcSharedCols') || '4');
    var wlMcWidgets    = {};
    var wlLivePrices   = {};   // { ticker: { price, prevClose, updatedAt } }
    var wlPriceTimer   = null;

    // ── KV Storage helpers ────────────────────────────────────────────────
    function kvGet(key) {
        return fetch(WL_PROXY + '?action=kv_get&key=' + encodeURIComponent(key))
            .then(function(r) { return r.json(); })
            .then(function(d) { return d.value; })
            .catch(function() { return null; });
    }

    function kvSet(key, value) {
        return fetch(WL_PROXY + '?action=kv_set&key=' + encodeURIComponent(key), {
            method: 'POST',
            body: typeof value === 'string' ? value : JSON.stringify(value)
        }).catch(function() {});
    }

    var wlPickerTicker = null;
    var wlPickerOpen   = false;
    var wlPickerBtn    = null;
    var wlPickerJustOpened = false;

    function wlGetLastList() {
        var last = localStorage.getItem(LS_WL_LAST_KEY);
        var all  = wlGetAll();
        // Validate it still exists
        if (last && all[last]) return last;
        // Fall back to first list in order
        var order = wlGetOrder();
        return order.length ? order[0] : null;
    }
    function wlSetLastList(name) {
        try { localStorage.setItem(LS_WL_LAST_KEY, name); } catch(e) {}
        kvSet('wl_last', name);
    }

    function wlGetAll() {
        try { return JSON.parse(localStorage.getItem(LS_WL_KEY) || '{}'); } catch(e) { return {}; }
    }
    function wlGetOrder() {
        try {
            var all = wlGetAll();
            var ord = JSON.parse(localStorage.getItem(LS_WL_ORD_KEY) || '[]');
            Object.keys(all).forEach(function(k){ if (ord.indexOf(k) === -1) ord.push(k); });
            return ord.filter(function(k){ return k in all; });
        } catch(e) { return Object.keys(wlGetAll()); }
    }
    function wlSaveAll(obj) {
        try { localStorage.setItem(LS_WL_KEY, JSON.stringify(obj)); } catch(e) {}
        kvSet('wl_data', JSON.stringify(obj));
    }
    function wlSaveOrder(arr) {
        try { localStorage.setItem(LS_WL_ORD_KEY, JSON.stringify(arr)); } catch(e) {}
        kvSet('wl_order', JSON.stringify(arr));
    }

    function wlAllTickers() {
        var all = wlGetAll();
        var set = new Set();
        Object.values(all).forEach(function(tickers){ tickers.forEach(function(t){ set.add(t); }); });
        return set;
    }

    // Refresh star buttons across visible stock rows
    function wlRefreshStars() {
        var last   = wlGetLastList();
        var all    = wlGetAll();
        var inLast = last && all[last] ? new Set(all[last]) : new Set();
        var inAny  = wlAllTickers();
        document.querySelectorAll('.wl-add-btn').forEach(function(btn) {
            var t = btn.getAttribute('data-ticker');
            var active = inLast.has(t);
            btn.classList.toggle('in-wl', active);
            btn.textContent = active ? '★' : '☆';
            btn.title = active ? 'Remove from "' + (last||'') + '"' : (last ? 'Add to "' + last + '"' : 'Add to watchlist');
        });
        document.querySelectorAll('.wl-pick-btn').forEach(function(btn) {
            var t = btn.getAttribute('data-ticker');
            btn.classList.toggle('in-wl', inAny.has(t));
        });
    }

    // Update only buttons for one ticker — safe to call while picker is open
    function wlRefreshTickerStar(ticker) {
        var last   = wlGetLastList();
        var all    = wlGetAll();
        var inLast = last && all[last] ? all[last].indexOf(ticker) !== -1 : false;
        var inAny  = false;
        Object.values(all).forEach(function(arr){ if (arr.indexOf(ticker) !== -1) inAny = true; });
        document.querySelectorAll('.wl-add-btn[data-ticker="' + ticker + '"]').forEach(function(btn) {
            btn.classList.toggle('in-wl', inLast);
            btn.textContent = inLast ? '★' : '☆';
            btn.title = inLast ? 'Remove from "' + (last||'') + '"' : (last ? 'Add to "' + last + '"' : 'Add to watchlist');
        });
        document.querySelectorAll('.wl-pick-btn[data-ticker="' + ticker + '"]').forEach(function(btn) {
            btn.classList.toggle('in-wl', inAny);
        });
    }

    // ── Watchlist sort state ─────────────────────────────────────────────
    var _wlSortColStored = localStorage.getItem('wlSortCol');
    var _wlSortCol = _wlSortColStored === null ? 'chgp' : (_wlSortColStored || null);  // 'sym'|'last'|'chg'|'chgp'|null
    var _wlSortDirStored = localStorage.getItem('wlSortDir');
    var _wlSortDir = _wlSortDirStored === null ? 'desc' : (_wlSortDirStored || 'asc'); // 'asc'|'desc'

    window.wlCycleSort = function(col) {
        if (_wlSortCol !== col) {
            _wlSortCol = col; _wlSortDir = col === 'chgp' ? 'desc' : 'asc';
        } else if (_wlSortDir === 'asc') {
            _wlSortDir = 'desc';
        } else {
            _wlSortCol = null; _wlSortDir = 'asc'; // reset
        }
        localStorage.setItem('wlSortCol', _wlSortCol || '');
        localStorage.setItem('wlSortDir', _wlSortDir);
        wlRender();
    };

    window.wlRender = function() {
        currentWlIndex = -1;
        var scroll   = document.getElementById('wl-list-scroll');
        var all      = wlGetAll();
        var order    = wlGetOrder();
        var active   = wlGetLastList();
        var nameEl   = document.getElementById('wl-selector-name');

        // Update selector label
        if (nameEl) nameEl.textContent = active || '— select list —';

        if (!order.length) {
            scroll.innerHTML = '<div class="wl-empty-state">No watchlists yet.<br>Add stocks using the ☆ button<br>on any stock row.</div>';
            return;
        }
        if (!active || !all[active]) {
            scroll.innerHTML = '<div class="wl-empty-state">Select a watchlist above.</div>';
            return;
        }

        var tickers = (all[active] || []).slice(); // copy so we don't mutate stored order
        var nameAttr = esc(active);
        var html = '';

        // Build per-ticker data map for sorting
        var tickerData = {};
        tickers.forEach(function(t) {
            var sd   = wlLookupStock(t);
            var live = wlLivePrices[t];
            var price, dayVal, chgAbs;
            if (live && live.price && live.prevClose) {
                price  = live.price;
                chgAbs = price - live.prevClose;
                dayVal = (chgAbs / live.prevClose) * 100;
            } else {
                price  = sd && sd.price != null ? sd.price : null;
                if (live && live.price && !price) price = live.price;
                dayVal = sd ? sd.daily : null;
                chgAbs = (price != null && dayVal != null) ? (price / (1 + dayVal / 100)) * (dayVal / 100) : null;
            }
            tickerData[t] = { price: price, chgAbs: chgAbs, dayVal: dayVal };
        });

        // Sort if a column is active
        if (_wlSortCol) {
            tickers.sort(function(a, b) {
                var av, bv;
                if (_wlSortCol === 'sym') {
                    av = a; bv = b;
                    return _wlSortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
                }
                var key = _wlSortCol === 'last' ? 'price' : _wlSortCol === 'chg' ? 'chgAbs' : 'dayVal';
                av = tickerData[a][key]; bv = tickerData[b][key];
                av = av != null ? av : (_wlSortDir === 'asc' ? Infinity : -Infinity);
                bv = bv != null ? bv : (_wlSortDir === 'asc' ? Infinity : -Infinity);
                return _wlSortDir === 'asc' ? av - bv : bv - av;
            });
        }

        function hdrSpan(cls, col, label) {
            var sorted = _wlSortCol === col;
            var arrow  = sorted ? (_wlSortDir === 'asc' ? ' ↑' : ' ↓') : '';
            return '<span class="' + cls + (sorted ? ' wl-c-sorted' : '') + '" onclick="wlCycleSort(\'' + col + '\')">' + label + arrow + '</span>';
        }

        if (!tickers.length) {
            html = '<div class="wl-empty-state">No stocks in this list yet.</div>';
        } else {
            html += '<div class="wl-col-hdr">';
            html += hdrSpan('wl-c-sym',  'sym',  'Symbol');
            html += hdrSpan('wl-c-last', 'last', 'Last');
            html += hdrSpan('wl-c-chg',  'chg',  'Chg');
            html += hdrSpan('wl-c-chgp', 'chgp', 'Chg%');
            html += '<span class="wl-c-del"></span>';
            html += '</div>';
            tickers.forEach(function(t) {
                var d = tickerData[t];
                var price = d.price, chgAbs = d.chgAbs, dayVal = d.dayVal;
                var isActive  = (wlChartTicker === t) ? ' active' : '';
                var priceStr  = price != null ? price.toFixed(2) : '—';
                var chgStr    = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '—';
                var chgpStr   = dayVal != null ? (dayVal >= 0 ? '+' : '') + dayVal.toFixed(2) + '%' : '—';
                var cl        = dayVal == null ? 'neutral' : dayVal > 0 ? 'up' : dayVal < 0 ? 'down' : 'neutral';
                html += '<div class="wl-ticker-row' + isActive + '" data-wl-ticker="' + esc(t) + '" data-wl-list="' + nameAttr + '">';
                html += '<span class="wl-c-sym">' + esc(t) + '</span>';
                html += '<span class="wl-c-last">' + priceStr + '</span>';
                html += '<span class="wl-c-chg ' + cl + '">' + chgStr + '</span>';
                html += '<span class="wl-c-chgp ' + cl + '">' + chgpStr + '</span>';
                html += '<button class="wl-ticker-remove" data-wl-remove-ticker="' + esc(t) + '" data-wl-remove-list="' + nameAttr + '" title="Remove">✕</button>';
                html += '</div>';
            });
        }
        scroll.innerHTML = html;
        if (typeof tickerHoverBind === 'function') tickerHoverBind(scroll, '.wl-c-sym', function(el) {
            var t = el.textContent.trim();
            return (t && t !== 'Symbol') ? t : null;
        });
        if (typeof alStampBadges === 'function') alStampBadges();
        if (wlMcActive) renderWlMc();
    };

    var wlSelectorOpen = false;

    window.wlToggleSelector = function() {
        var dd    = document.getElementById('wl-selector-dropdown');
        var all   = wlGetAll();
        var order = wlGetOrder();
        var active = wlGetLastList();

        if (wlSelectorOpen) {
            dd.style.display = 'none';
            wlSelectorOpen = false;
            return;
        }

        var html = '';
        order.forEach(function(name) {
            var count    = (all[name] || []).length;
            var isActive = name === active;
            html += '<div class="wl-selector-item' + (isActive ? ' active' : '') + '" data-wl-switch="' + esc(name) + '">';
            html += '<span class="wl-selector-item-check">' + (isActive ? '✓' : '') + '</span>';
            html += esc(name);
            html += '<span class="wl-selector-item-right">';
            html += '<span class="wl-selector-item-count">' + count + '</span>';
            html += '<button class="wl-selector-item-clear" data-wl-clear-list="' + esc(name) + '" title="Clear all tickers">⊘</button>';
            html += '<button class="wl-selector-item-del" data-wl-del-list="' + esc(name) + '" title="Delete">✕</button>';
            html += '</span>';
            html += '</div>';
        });

        dd.innerHTML = html;
        dd.style.display = 'block';
        wlSelectorOpen = true;

        // Delegate clicks inside dropdown
        dd.onclick = function(e) {
            var clearBtn = e.target.closest('[data-wl-clear-list]');
            if (clearBtn) {
                e.stopPropagation();
                var name = clearBtn.getAttribute('data-wl-clear-list');
                var rightEl = clearBtn.closest('.wl-selector-item-right');
                rightEl.innerHTML =
                    '<span style="font-size:0.78em;color:#8b949e;margin-right:4px;">Clear?</span>' +
                    '<button class="wl-selector-item-confirm-yes" style="background:none;border:1px solid #3fb950;border-radius:3px;color:#3fb950;font-size:0.72em;padding:1px 6px;cursor:pointer;" data-wl-confirm-clear="' + esc(name) + '">Yes</button>' +
                    '<button class="wl-selector-item-confirm-no" style="background:none;border:1px solid #484f58;border-radius:3px;color:#6e7681;font-size:0.72em;padding:1px 6px;cursor:pointer;">No</button>';
                return;
            }
            var confirmYes = e.target.closest('[data-wl-confirm-clear]');
            if (confirmYes) {
                e.stopPropagation();
                var name = confirmYes.getAttribute('data-wl-confirm-clear');
                var all = wlGetAll();
                all[name] = [];
                wlSaveAll(all);
                wlToggleSelector();
                wlRender();
                wlRefreshStars();
                return;
            }
            var confirmNo = e.target.closest('.wl-selector-item-confirm-no');
            if (confirmNo) {
                e.stopPropagation();
                wlToggleSelector();
                wlToggleSelector();
                return;
            }
            var delBtn = e.target.closest('[data-wl-del-list]');
            if (delBtn) {
                e.stopPropagation();
                var name = delBtn.getAttribute('data-wl-del-list');
                var rightEl = delBtn.closest('.wl-selector-item-right');
                rightEl.innerHTML =
                    '<span style="font-size:0.78em;color:#8b949e;margin-right:4px;">Delete?</span>' +
                    '<button style="background:none;border:1px solid #f85149;border-radius:3px;color:#f85149;font-size:0.72em;padding:1px 6px;cursor:pointer;" data-wl-confirm-del="' + esc(name) + '">Yes</button>' +
                    '<button class="wl-selector-item-confirm-no" style="background:none;border:1px solid #484f58;border-radius:3px;color:#6e7681;font-size:0.72em;padding:1px 6px;cursor:pointer;">No</button>';
                return;
            }
            var confirmDel = e.target.closest('[data-wl-confirm-del]');
            if (confirmDel) {
                e.stopPropagation();
                var name = confirmDel.getAttribute('data-wl-confirm-del');
                var all = wlGetAll();
                delete all[name];
                wlSaveAll(all);
                wlSaveOrder(wlGetOrder().filter(function(k){ return k !== name; }));
                dd.style.display = 'none';
                wlSelectorOpen = false;
                wlRender();
                wlRefreshStars();
                return;
            }
            var item = e.target.closest('[data-wl-switch]');
            if (!item) return;
            var name = item.getAttribute('data-wl-switch');
            wlSetLastList(name);
            wlLivePrices = {};
            dd.style.display = 'none';
            wlSelectorOpen = false;
            wlRender();
            wlRefreshStars();
            wlStartPricePolling();
        };
    };

    // Close selector on outside click
    document.addEventListener('click', function(e) {
        if (!wlSelectorOpen) return;
        var dd  = document.getElementById('wl-selector-dropdown');
        var sel = document.getElementById('wl-selector');
        if (dd && !dd.contains(e.target) && sel && !sel.contains(e.target)) {
            dd.style.display = 'none';
            wlSelectorOpen = false;
        }
    });

    // Look up a stock's data from snapshot across all industries
    function wlLookupStock(ticker) {
        if (!snapshot || !snapshot.by_industry) return null;
        var industries = snapshot.by_industry;
        for (var ind in industries) {
            var rows = industries[ind];
            for (var i = 0; i < rows.length; i++) {
                if (rows[i].ticker === ticker) return rows[i];
            }
        }
        return null;
    }

    // Single delegated listener on the scroll container — registered once at startup
    window.wlToggleExport = function(e) {
        e.stopPropagation();
        var dd = document.getElementById('wl-export-dropdown');
        dd.style.display = dd.style.display === 'none' ? 'block' : 'none';
    };

    document.addEventListener('click', function() {
        var dd = document.getElementById('wl-export-dropdown');
        if (dd) dd.style.display = 'none';
    });

    window.wlExport = function(format) {
        document.getElementById('wl-export-dropdown').style.display = 'none';
        var name = document.getElementById('wl-selector-name').textContent || 'watchlist';
        var all  = wlGetAll();
        var stocks = all[name] || [];

        if (!stocks.length) { alert('No stocks in this watchlist.'); return; }

        if (format === 'tickers') {
            var text = stocks.join(',');
            var ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            // Brief visual feedback
            var btn = document.querySelector('#wl-export-dropdown').previousElementSibling;
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(function(){ btn.textContent = orig; }, 1500);
            return;
        }

        if (format === 'csv') {
            // Build CSV with snapshot data if available
            var rows = ['Symbol,Price,Change%,RS,Industry'];
            stocks.forEach(function(ticker) {
                var data = null;
                if (snapshot && snapshot.by_industry) {
                    Object.values(snapshot.by_industry).forEach(function(arr) {
                        arr.forEach(function(r) { if (r.ticker === ticker) data = r; });
                    });
                }
                if (data) {
                    rows.push([
                        ticker,
                        data.price != null ? data.price.toFixed(2) : '',
                        data.daily != null ? data.daily.toFixed(2) : '',
                        data.Percentile != null ? Math.round(data.Percentile) : '',
                        '"' + (data.industry || '') + '"'
                    ].join(','));
                } else {
                    rows.push([ticker,'','','',''].join(','));
                }
            });
            var blob = new Blob([rows.join('\n')], { type: 'text/csv' });
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = name.replace(/[^a-z0-9]/gi,'_') + '.csv';
            a.click();
        }
    };

    window.wlImportCSV = function() {
        document.getElementById('wl-export-dropdown').style.display = 'none';
        var fileInput = document.getElementById('wl-import-file');
        fileInput.value = '';
        fileInput.click();
    };

    document.getElementById('wl-import-file').addEventListener('change', function(e) {
        var file = e.target.files && e.target.files[0];
        if (!file) return;

        // Derive list name from filename: strip .csv, underscores → spaces
        var listName = file.name.replace(/\.csv$/i, '').replace(/_/g, ' ').trim();
        if (!listName) listName = 'Imported';

        var reader = new FileReader();
        reader.onload = function(ev) {
            var lines = (ev.target.result || '').split(/\r?\n/);
            var tickers = [];
            lines.forEach(function(line, idx) {
                if (!line.trim()) return;
                var first = line.split(',')[0].trim().replace(/^"|"$/g, '').toUpperCase();
                if (idx === 0 && first === 'SYMBOL') return; // skip header
                if (first) tickers.push(first);
            });

            if (!tickers.length) { alert('No symbols found in CSV.'); return; }

            var all = wlGetAll();
            if (!all[listName]) all[listName] = [];

            // Merge — add only tickers not already in the list
            var added = 0;
            tickers.forEach(function(t) {
                if (all[listName].indexOf(t) === -1) {
                    all[listName].push(t);
                    added++;
                }
            });

            // Ensure list appears in order
            var order = wlGetOrder();
            if (order.indexOf(listName) === -1) {
                order.push(listName);
                wlSaveOrder(order);
            }

            wlSaveAll(all);
            wlRender();
            alert('Imported ' + added + ' ticker' + (added !== 1 ? 's' : '') + ' into "' + listName + '".' + (tickers.length - added > 0 ? '\n(' + (tickers.length - added) + ' already present, skipped)' : ''));
        };
        reader.readAsText(file);
    });

    function wlAttachListeners() {
        var scroll = document.getElementById('wl-list-scroll');
        if (!scroll) return;
        scroll.addEventListener('contextmenu', function(e) {
            var tickerRow = e.target.closest('[data-wl-ticker]');
            if (!tickerRow) return;
            e.preventDefault();
            var ticker = tickerRow.getAttribute('data-wl-ticker');
            var fakeBtn = {
                getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
                getBoundingClientRect: function() {
                    return { bottom: e.clientY, top: e.clientY, left: e.clientX };
                },
                _wlNoSwitch: true
            };
            wlOpenPicker(fakeBtn, e, false);
        });

        scroll.addEventListener('click', function(e) {
            // Remove ticker from list
            var remBtn = e.target.closest('[data-wl-remove-ticker]');
            if (remBtn) {
                e.stopPropagation();
                var ticker   = remBtn.getAttribute('data-wl-remove-ticker');
                var listName = remBtn.getAttribute('data-wl-remove-list');
                var all = wlGetAll();
                if (!all[listName]) return;
                all[listName] = all[listName].filter(function(t){ return t !== ticker; });
                wlSaveAll(all);
                wlRender();
                wlRefreshStars();
                return;
            }
            // Select ticker for chart
            var tickerRow = e.target.closest('[data-wl-ticker]');
            if (tickerRow && !e.target.closest('[data-wl-remove-ticker]')) {
                var wlRows = Array.from(document.querySelectorAll('.wl-ticker-row'));
                currentWlIndex = wlRows.indexOf(tickerRow);
                wlSelectTicker(tickerRow.getAttribute('data-wl-ticker'));
            }
        });
    }

    window.wlCreateNew = function() {
        var row   = document.getElementById('wl-new-row');
        var input = document.getElementById('wl-new-input');
        row.style.display = 'flex';
        input.value = '';
        input.focus();
    };

    window.wlConfirmNew = function() {
        var input = document.getElementById('wl-new-input');
        var name  = input.value.trim();
        if (!name) { input.focus(); return; }
        var all = wlGetAll();
        if (all[name]) {
            input.style.borderBottomColor = '#f85149';
            input.select();
            setTimeout(function(){ input.style.borderBottomColor = '#388bfd'; }, 1200);
            return;
        }
        all[name] = [];
        wlSaveAll(all);
        var order = [];
        try { order = JSON.parse(localStorage.getItem(LS_WL_ORD_KEY) || '[]'); } catch(e) {}
        order = order.filter(function(k){ return k in all; });
        if (order.indexOf(name) === -1) order.push(name);
        wlSaveOrder(order);
        wlSetLastList(name);
        // Open new list accordion-style (close all others)
        wlCancelNew();
        wlRender();
    };

    window.wlCancelNew = function() {
        var row = document.getElementById('wl-new-row');
        row.style.display = 'none';
    };

    window.wlOpenAddSym = function() {
        var row   = document.getElementById('wl-add-sym-row');
        var input = document.getElementById('wl-add-sym-input');
        var btnRow = document.getElementById('wl-add-sym-btn-row');
        row.style.display = 'flex';
        btnRow.style.display = 'none';
        input.value = '';
        input.focus();
    };

    window.wlConfirmAddSym = function() {
        var input  = document.getElementById('wl-add-sym-input');
        var raw    = input.value.trim().toUpperCase();
        if (!raw) { input.focus(); return; }
        var active = wlGetLastList();
        if (!active) { wlCancelAddSym(); return; }
        var all = wlGetAll();
        if (!all[active]) all[active] = [];
        var tickers = raw.split(/[\s,]+/).filter(function(t) { return t.length > 0; });
        tickers.forEach(function(ticker) {
            if (all[active].indexOf(ticker) === -1) all[active].push(ticker);
        });
        wlSaveAll(all);
        wlCancelAddSym();
        wlRender();
        wlRefreshStars();
    };

    window.wlCancelAddSym = function() {
        document.getElementById('wl-add-sym-row').style.display = 'none';
        document.getElementById('wl-add-sym-btn-row').style.display = 'block';
    };

    // Keyboard handling for add-symbol input
    document.addEventListener('keydown', function(e) {
        var symRow = document.getElementById('wl-add-sym-row');
        if (!symRow || symRow.style.display === 'none') return;
        if (document.activeElement === document.getElementById('wl-add-sym-input')) {
            if (e.key === 'Enter')  { e.preventDefault(); wlConfirmAddSym(); }
            if (e.key === 'Escape') { e.preventDefault(); wlCancelAddSym(); }
        }
    });

    // Keyboard handling for the inline input
    document.addEventListener('keydown', function(e) {
        var row = document.getElementById('wl-new-row');
        if (!row || row.style.display === 'none') return;
        if (document.activeElement === document.getElementById('wl-new-input')) {
            if (e.key === 'Enter')  { e.preventDefault(); wlConfirmNew(); }
            if (e.key === 'Escape') { e.preventDefault(); wlCancelNew(); }
        }
    });

    window.wlDeleteList = function(name) {
        if (!confirm('Delete watchlist "' + name + '"?')) return;
        var all = wlGetAll();
        delete all[name];
        wlSaveAll(all);
        var order = wlGetOrder().filter(function(k){ return k !== name; });
        wlSaveOrder(order);
        delete wlOpenGroups[name];
        wlRender();
        wlRefreshStars();
    };

    window.wlRemoveTicker = function(name, ticker) {
        var all = wlGetAll();
        if (!all[name]) return;
        all[name] = all[name].filter(function(t){ return t !== ticker; });
        wlSaveAll(all);
        if (wlChartTicker === ticker) {
            // keep chart open, just update active state
        }
        wlRender();
        wlRefreshStars();
    };

    window.wlAddTicker = function(name, ticker) {
        var all = wlGetAll();
        if (!all[name]) all[name] = [];
        if (all[name].indexOf(ticker) === -1) all[name].push(ticker);
        wlSaveAll(all);
        wlRender();
        wlRefreshStars();
    };

    window.wlSelectTicker = function(ticker) {
        wlChartTicker = ticker;

        // Update header — symbol, industry · rank, fundamentals/news links
        document.getElementById('wl-chart-sym').textContent = ticker;
        updateQueueButtons();
        var metaEl = document.getElementById('wl-chart-meta');
        var fBtn   = document.getElementById('wl-chart-details-btn');
        var t = ticker.replace(/[^A-Z0-9]/gi, '');
        if (fBtn) { fBtn.href = 'https://finviz.com/quote.ashx?t=' + t; fBtn.style.display = ''; }

        // Populate industry · rank from snapshot/industriesData
        var industry = '', sector = '', wlPct = null, wlFundRow = null;
        if (snapshot && snapshot.by_industry) {
            outer: for (var ind in snapshot.by_industry) {
                var rows = snapshot.by_industry[ind];
                for (var i = 0; i < rows.length; i++) {
                    if (rows[i].ticker === ticker) {
                        industry  = rows[i].industry || '';
                        sector    = rows[i].sector   || '';
                        wlPct     = rows[i].Percentile != null ? rows[i].Percentile : null;
                        wlFundRow = rows[i];
                        break outer;
                    }
                }
            }
        }
        if (metaEl) {
            var indRankHtml = '';
            if (industry && industriesData && industriesData.industries) {
                var indData = industriesData.industries.find(function(x){ return x.industry === industry; });
                var total   = industriesData.industries.length;
                if (indData && indData.rank != null) {
                    var pct = indData.percentile != null ? indData.percentile : null;
                    var rankColor = pct != null ? (pct >= 75 ? '#3fb950' : pct >= 40 ? '#e3852b' : '#f85149') : '#6e7681';
                    indRankHtml = '<span style="margin:0 5px;color:#30363d;">·</span>' +
                        '<span style="color:' + rankColor + '">(' + indData.rank + '/' + total + ')</span>';
                }
            }
            metaEl.innerHTML = industry ? industryLinkHtml(industry, null) + indRankHtml : '';
        }
        applyRsBadge(document.getElementById('wl-chart-rs-badge'), wlPct, wlFundRow ? wlFundRow.weighted_rs_pct : null, document.getElementById('wl-chart-3mrs-badge'));
        var wlFundStatsEl = document.getElementById('wl-chart-fund-stats');
        if (wlFundStatsEl) wlFundStatsEl.innerHTML = fundStatsHtml(wlFundRow);

        // Show LW chart — destroy any previous instance first
        document.getElementById('wl-chart-empty').style.display = 'none';
        var widgetDiv = document.getElementById('wl-chart-widget');
        widgetDiv.style.display = 'block';

        // Show settings bar and sync TF buttons to current wlChartTf
        var settingsBar = document.getElementById('wl-chart-settings');
        if (settingsBar) settingsBar.style.display = 'flex';
        document.querySelectorAll('.wl-chart-fs-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === wlChartTf);
        });

        // Reset per-symbol tool state
        _wlVwapMode = false; _wlVwapSeries = []; _wlSelectedVwapIdx = -1;
        var vwapBtn = document.getElementById('wl-chart-vwap-btn');
        if (vwapBtn) vwapBtn.classList.remove('active');
        _wlTrendlines = []; _wlTrendlineFirst = null; _wlSelectedTrendlineIdx = -1;
        if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
        _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
        var maPanel   = document.getElementById('wl-chart-ma-panel');
        var maChevron = document.getElementById('wl-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';

        _wlVisibleBars = wlChartTf === 'D' ? 252 : wlChartTf === 'W' ? 104 : 60;
        widgetDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">Loading…</div>';
        var loadTicker = ticker;
        fetchMcOhlcv(ticker, wlChartTf).then(function(ohlcv) {
            // Guard: a newer ticker was already selected
            if (wlChartTicker !== loadTicker) return;
            _buildWlChart(loadTicker, ohlcv, wlChartTf);
        });
        // Update active row highlight
        document.querySelectorAll('.wl-ticker-row').forEach(function(r){
            r.classList.toggle('active', r.getAttribute('data-wl-ticker') === ticker);
        });
    };

    // ── Watchlist picker popover ──────────────────────────────────────────

    // Quick-add: star click adds/removes from last-used list immediately
    window.wlQuickToggle = function(btn) {
        var ticker = btn.getAttribute('data-ticker');
        var last   = wlGetLastList();
        if (!last) {
            // No list yet — fall through to picker
            wlOpenPicker(btn, { stopPropagation: function(){} });
            return;
        }
        var all = wlGetAll();
        if (!all[last]) all[last] = [];
        var idx = all[last].indexOf(ticker);
        if (idx !== -1) {
            all[last].splice(idx, 1);
        } else {
            all[last].push(ticker);
        }
        wlSaveAll(all);
        if (currentView === 'watchlists') wlRender();
        wlRefreshStars();
    };

    window.wlOpenPicker = function(btn, e, internal) {
        e.stopPropagation();
        // If this exact button already has the picker open, close it
        if (!internal && wlPickerOpen && wlPickerBtn === btn) {
            wlClosePicker();
            return;
        }
        // btn may be the ▾ button — get ticker from it directly (both star and chevron have data-ticker)
        var ticker = btn.getAttribute('data-ticker');
        wlPickerTicker = ticker;
        wlPickerBtn    = btn;

        var all    = wlGetAll();
        var order  = wlGetOrder();
        var picker = document.getElementById('wl-picker');

        var html = '<div class="wl-picker-title">Add to watchlist<button class="wl-picker-close" title="Close">✕</button></div>';

        if (order.length) {
            order.forEach(function(name) {
                var has      = all[name] && all[name].indexOf(ticker) !== -1;
                html += '<div class="wl-picker-item' + (has ? ' checked' : '') + '" data-wl-pick="' + esc(name) + '">';
                html += '<span class="wl-picker-check">' + (has ? '✓' : '') + '</span>';
                html += esc(name);
                html += '</div>';
            });
            html += '<div class="wl-picker-divider"></div>';
        }

        html += '<div class="wl-picker-new" id="wl-picker-new-btn">';
        html += '<span style="font-size:1em;color:#6e7681;">+</span> New watchlist…';
        html += '</div>';
        html += '<div class="wl-picker-divider"></div>';
        html += '<div class="wl-picker-new" id="wl-picker-alert-btn" style="color:#58a6ff;">';
        html += '<span style="font-size:1em;color:#6e7681;">+</span> Add Alert';
        html += '</div>';

        picker.innerHTML = html;

        picker.onclick = function(ev) {
            ev.stopPropagation();
            if (ev.target.closest('.wl-picker-close')) { wlClosePicker(); return; }
            var item = ev.target.closest('[data-wl-pick]');
            if (item) { wlPickerToggle(item.getAttribute('data-wl-pick')); return; }
            var newBtn = ev.target.closest('#wl-picker-new-btn');
            if (newBtn) { wlPickerNewList(); return; }
            var alertBtn = ev.target.closest('#wl-picker-alert-btn');
            if (alertBtn) {
                // Capture ticker immediately before wlClosePicker() nulls wlPickerTicker
                // Capture ticker NOW before anything is closed/nulled
                var _alertTicker = ticker || wlPickerTicker;
                wlClosePicker();

                // Open the alert modal — works from any view, no navigation needed
                alShowForm(_alertTicker);
                return;
            }
        };

        picker.style.visibility = 'hidden';
        picker.style.display = 'block';

        var rect = btn.getBoundingClientRect();
        var ph = picker.offsetHeight;
        var top = rect.bottom + 4;
        if (top + ph > window.innerHeight) top = rect.top - ph - 4;
        if (top < 4) top = 4;
        picker.style.top  = top + 'px';
        picker.style.left = Math.min(rect.left, window.innerWidth - 250) + 'px';
        picker.style.visibility = '';
        var bd = document.getElementById('wl-picker-backdrop');
        if (bd) bd.classList.add('open');
        wlPickerOpen = true;
        if (!internal) wlPickerJustOpened = true;
    };

    window.wlPickerToggle = function(name) {
        var ticker = wlPickerTicker;
        var btn    = wlPickerBtn;
        if (!ticker) return;
        var all = wlGetAll();
        if (!all[name]) all[name] = [];
        var idx = all[name].indexOf(ticker);
        if (idx !== -1) {
            all[name].splice(idx, 1);
        } else {
            all[name].push(ticker);
            if (!btn || !btn._wlNoSwitch) wlSetLastList(name);
        }
        wlSaveAll(all);
        if (currentView === 'watchlists') wlRender();
        // Re-open picker using same btn reference — do NOT call wlRefreshStars here
        // because it rebuilds innerHTML and destroys btn
        if (btn && btn.nodeType && document.contains(btn)) {
            wlOpenPicker(btn, { stopPropagation: function(){} }, true);
        } else if (btn && btn._wlNoSwitch) {
            // Fake button from right-click — re-open picker at same position
            wlOpenPicker(btn, { stopPropagation: function(){} }, true);
        } else {
            wlClosePicker();
        }
        // Only update the star for this specific ticker — avoids rebuilding rows
        wlRefreshTickerStar(ticker);
    };

    window.wlPickerNewList = function() {
        var picker = document.getElementById('wl-picker');
        var newRow = picker.querySelector('.wl-picker-new');
        if (!newRow) return;
        newRow.outerHTML = '<input class="wl-picker-input" id="wl-picker-input" type="text" placeholder="List name…" maxlength="40">';
        var input = document.getElementById('wl-picker-input');
        input.focus();
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                var name = input.value.trim();
                if (!name) return;
                var all = wlGetAll();
                if (!all[name]) {
                    all[name] = [];
                    // Read raw order without triggering wlGetOrder's auto-add logic
                    var order = [];
                    try { order = JSON.parse(localStorage.getItem(LS_WL_ORD_KEY) || '[]'); } catch(ex) {}
                    if (order.indexOf(name) === -1) order.push(name);
                    wlSaveOrder(order);
                }
                if (wlPickerTicker && all[name].indexOf(wlPickerTicker) === -1) {
                    all[name].push(wlPickerTicker);
                    wlSetLastList(name);
                }
                wlSaveAll(all);
                wlRefreshStars();
                if (currentView === 'watchlists') wlRender();
                wlClosePicker();
            }
            if (e.key === 'Escape') wlClosePicker();
        });
    };

    window.wlClosePicker = function wlClosePicker() {
        document.getElementById('wl-picker').style.display = 'none';
        var bd = document.getElementById('wl-picker-backdrop');
        if (bd) bd.classList.remove('open');
        wlPickerOpen = false;
        wlPickerTicker = null;
        wlPickerBtn = null;
    }

    // Close picker on Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && wlPickerOpen) wlClosePicker();
    });

    // When the picker backdrop intercepts a right-click, close the picker and
    // re-dispatch the contextmenu event onto whatever element is underneath so
    // handlers (e.g. alert-list rows) still fire on the second right-click.
    (function() {
        var bd = document.getElementById('wl-picker-backdrop');
        if (!bd) return;
        bd.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            wlClosePicker();
            bd.classList.remove('open');  // hide immediately so elementFromPoint skips it
            var el = document.elementFromPoint(e.clientX, e.clientY);
            if (el) {
                el.dispatchEvent(new MouseEvent('contextmenu', {
                    bubbles: true, cancelable: true,
                    clientX: e.clientX, clientY: e.clientY,
                    screenX: e.screenX, screenY: e.screenY
                }));
            }
        });
    }());

