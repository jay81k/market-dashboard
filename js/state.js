    // ── State ──────────────────────────────────────────────────────────────
    var snapshot      = null;
    var tickerMap     = {};
    var industriesData= null;
    var indPrevRanks  = {};   // { "Industry Name": rank } from previous day's snapshot
    var currentView   = 'industries';
    var _lastIndustryName        = null;
    var _lastIndustryScrollTop   = 0;
    var _industriesListScrollTop      = 0;
    var _indStocksScrollBeforeModal   = 0;
    var activeSort    = 'rank';
    var indSort       = { col: null, dir: -1 }; // dir: -1 = desc, 1 = asc
    var searchQuery   = '';
    var currentStockSort = { by: null, dir: 1, count: 0 };
    var activeMAType   = localStorage.getItem('distMAType')   || 'SMA';
    var activeMALength = parseInt(localStorage.getItem('distMALength') || '50');
    var allStockRows   = [];
    var currentStockIndex = -1;
    var allIndustryRows = [];
    var currentIndustryIndex = -1;
    var currentWlIndex = -1;
    var WL_PROXY = 'https://yahoo-proxy.jay69k.workers.dev';

    // ── Live intraday Day% for industry list + heatmap ────────────────────
    var _indLiveDayInterval = null;

    function fetchLiveIndustryDay() {
        if (!snapshot || !snapshot.by_industry || !industriesData || !industriesData.industries) return;

        // Build ticker → industry map and collect all tickers
        var tickerToInd = {};
        Object.keys(snapshot.by_industry).forEach(function(indName) {
            snapshot.by_industry[indName].forEach(function(row) {
                if (row.ticker) tickerToInd[row.ticker] = indName;
            });
        });
        var allTickers = Object.keys(tickerToInd);
        if (!allTickers.length) return;

        // Accumulator: { industryName: { sum: 0, count: 0 } }
        var acc = {};
        industriesData.industries.forEach(function(ind) { acc[ind.industry] = { sum: 0, count: 0 }; });

        // Batch into 30s — matches the Worker's cap (a 50-ticker batch measurably
        // exceeded its 10ms CPU budget).
        var batches = [];
        for (var i = 0; i < allTickers.length; i += 30) batches.push(allTickers.slice(i, i + 30));

        var pending = batches.length;
        if (!pending) return;

        // Fires through the same shared clock as multichart.js/market.js
        // (yahoo-proxy-pace.js) instead of a fixed timer — a burst here can
        // trip the same upstream Yahoo limit those scripts are also hitting,
        // so all three now back off together.
        //
        // Caveat: this only paces WHEN each batch *request* fires. It can't
        // smooth out what happens inside the Worker during that request —
        // quotes_batch fetches all ~30 of a batch's tickers from Yahoo
        // concurrently, server-side, invisible to any client-side pacer. If
        // 429s persist after this, that's a separate, Worker-side fix
        // (capping concurrency inside quotes_batch itself), not something
        // fixable from here.
        var STATE_LAUNCH_MIN_SPACING = 120;
        var queue = batches.slice();

        function launchNextBatch() {
            if (queue.length === 0) return;

            var wait      = Math.max(0, window.yahooProxyPace.cooldownUntil() - Date.now());
            var sinceLast = Date.now() - window.yahooProxyPace.lastLaunchAt();
            if (sinceLast < STATE_LAUNCH_MIN_SPACING) wait = Math.max(wait, STATE_LAUNCH_MIN_SPACING - sinceLast);

            if (wait > 0) { setTimeout(launchNextBatch, wait + 5); return; }

            var batch = queue.shift();
            window.yahooProxyPace.markLaunched();
            var url = WL_PROXY + '?action=quotes_batch&tickers=' + batch.map(encodeURIComponent).join(',');
            fetch(url).then(function(r) {
                if (r.status === 429) window.yahooProxyPace.register429();
                return r.ok ? r.json() : null;
            }).then(function(data) {
                if (data && data.quotes) {
                    data.quotes.forEach(function(q) {
                        if (!q || !q.ticker || !q.price || !q.prevClose || q.prevClose <= 0) return;
                        var indName = tickerToInd[q.ticker];
                        if (!indName || !acc[indName]) return;
                        var pct = ((q.price - q.prevClose) / q.prevClose) * 100;
                        acc[indName].sum   += pct;
                        acc[indName].count += 1;
                        // Write live price & daily % back so renderMarketMovers() sees fresh data
                        var row = tickerMap[q.ticker];
                        if (row) { row.price = q.price; row.daily = pct; }
                    });
                }
            }).catch(function() {}).finally(function() {
                pending--;
                if (pending === 0) _applyLiveIndustryDay(acc);
            });

            launchNextBatch(); // launch the next one too — its own turn re-checks spacing/cooldown
        }

        launchNextBatch();
    }

    function _applyLiveIndustryDay(acc) {
        // Patch avg_daily in both industriesData and snapshot.industry_summary
        // (renderers read from snapshot.industry_summary, so that must be updated)
        industriesData.industries.forEach(function(ind) {
            var a = acc[ind.industry];
            if (a && a.count > 0) {
                var liveVal = parseFloat((a.sum / a.count).toFixed(4));
                ind.avg_daily = liveVal; // keep in sync for sort paths
                if (snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]) {
                    snapshot.industry_summary[ind.industry].avg_daily = liveVal;
                }
            }
        });
        // Re-render only if user is on the industry or market view
        if (currentView === 'market') { renderMarketMovers(); renderMarketBreadth(); }
        if (currentView !== 'industries' && currentView !== 'sector') return;
        if (indView === 'heatmap') renderHeatmap();
        else renderIndustries();
    }
    // ── END Live intraday Day% ─────────────────────────────────────────────

    // ── Ticker Queue (copy list) ───────────────────────────────────────────
    var tickerQueue = [];

    var QUEUE_BTN_IDS = ['wl-chart-queue-btn', 'mc-fullscreen-queue-btn'];

    window.toggleTickerQueue = function(ticker) {
        ticker = (ticker || '').trim();
        if (!ticker || ticker === '—') return;
        var idx = tickerQueue.indexOf(ticker);
        if (idx === -1) {
            tickerQueue.push(ticker);
        } else {
            tickerQueue.splice(idx, 1);
        }
        updateQueueButtons();
        copyTickerQueueSilent();
    };

    function updateQueueButtons() {
        QUEUE_BTN_IDS.forEach(function(id) {
            var btn = document.getElementById(id);
            if (!btn) return;
            var symId = id.replace('-queue-btn', '-sym');
            var sym = document.getElementById(symId);
            var ticker = sym ? sym.textContent.trim() : '';
            var inQueue = tickerQueue.indexOf(ticker) !== -1;
            btn.classList.toggle('added', inQueue);
            btn.innerHTML = inQueue
                ? '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/></svg> ✓'
                : '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M5 0h6l3 3v10a1 1 0 01-1 1H3a1 1 0 01-1-1V1a1 1 0 011-1zm1 1v3H3v9h10V4h-2V1H6zm0 6h4v1H6V7zm0 2h4v1H6V9z"/></svg> +';
        });
    }

    function copyTickerQueueSilent() {
        if (!tickerQueue.length) return;
        var text = tickerQueue.join(',');

        function fallback() {
            var ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.top = '0'; ta.style.left = '0';
            ta.style.width = '2em'; ta.style.height = '2em';
            ta.style.padding = '0'; ta.style.border = 'none';
            ta.style.outline = 'none'; ta.style.boxShadow = 'none';
            ta.style.background = 'transparent';
            document.body.appendChild(ta);
            ta.focus(); ta.select();
            try { document.execCommand('copy'); } catch(e) {}
            document.body.removeChild(ta);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).catch(fallback);
        } else {
            fallback();
        }
    }

    window.clearTickerQueue = function() {
        tickerQueue = [];
        updateQueueButtons();
    };

    // ── Helpers ───────────────────────────────────────────────────────────
    function esc(s) {
        if (s == null) return '';
        return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function fmt(v, d, sfx) {
        if (v == null) return '<span style="color:#30363d">—</span>';
        return (v >= 0 ? '+' : '') + v.toFixed(d) + (sfx || '');
    }
    function cc(v) { return v == null ? '' : v > 0 ? 'up' : v < 0 ? 'down' : ''; }

    function sectorClass(sector) {
        if (!sector) return '';
        var s = sector.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
        return 'sector-' + s;
    }


    // ── View switching ─────────────────────────────────────────────────────
    window.showView = function(view) {
        if (currentView === 'industry-stocks') {
            var _indWrap = document.querySelector('#view-industry-stocks .stocks-table-wrap');
            if (_indWrap) _lastIndustryScrollTop = _indWrap.scrollTop;
        }
        currentView = view;
        currentIndustryIndex = -1;
        currentStockIndex    = -1;
        currentWlIndex       = -1;
        document.getElementById('view-industries').style.display      = 'none';
        document.getElementById('view-industry-stocks').classList.remove('active');
        document.getElementById('view-scans').style.display           = 'none';
        document.getElementById('view-sector').style.display          = 'none';
        document.getElementById('view-watchlists').style.display      = 'none';
        document.getElementById('view-market').style.display          = 'none';
        document.getElementById('view-alerts').style.display          = 'none';
        if (view !== 'market') marketStopTimer();
        if (window._wlChartMsgHandler) { window.removeEventListener('message', window._wlChartMsgHandler); window._wlChartMsgHandler = null; }
        if (view !== 'watchlists') {
            wlStopPricePolling();
        }
        if (view !== 'scans')           scanStopPricePolling();
        if (view !== 'industry-stocks') indStopPricePolling();
        // Drops any tickers still queued from whatever grid view we're
        // leaving — every view transition runs through here, so this covers
        // Market/Industries/Scans/Watchlists in one place instead of each
        // view's own entry point having to remember to do it.
        if (window.mcClearGridQueue) mcClearGridQueue();
        if (view !== 'watchlists' && wlMcActive) {
            wlMcActive = false;
            var btn = document.getElementById('wl-multichart-toggle-btn');
            if (btn) { btn.style.background = ''; btn.style.borderColor = ''; btn.style.color = ''; }
            var mcView = document.getElementById('wl-multichart-view');
            if (mcView) mcView.style.display = 'none';
            var wlHdr = document.querySelector('.wl-chart-panel-header');
            if (wlHdr) wlHdr.style.display = '';
            wlMcWidgets = {};
        }

        document.getElementById('main-area').style.overflow = (view === 'watchlists') ? 'hidden' : '';

        document.getElementById('tab-industries').classList.toggle('active', view === 'industries' || view === 'industry-stocks' || view === 'sector');
        document.getElementById('tab-scans').classList.toggle('active', view === 'scans');
        document.getElementById('tab-watchlists').classList.toggle('active', view === 'watchlists');
        document.getElementById('tab-market').classList.toggle('active', view === 'market');
        document.getElementById('tab-alerts').classList.toggle('active', view === 'alerts');

        var searchEl = document.getElementById('search-input');
        searchEl.style.display = (view === 'market' || view === 'alerts') ? 'none' : '';
        var _scb = document.getElementById('search-clear-btn');
        if (_scb) _scb.classList.remove('visible');
        var _sw = searchEl.closest('.search-wrap');
        if (_sw) _sw.style.display = (view === 'market' || view === 'alerts') ? 'none' : '';
        searchEl.value = '';
        searchQuery = '';
        window._scansSearchQuery = '';
        if (view === 'industries') {
            searchEl.placeholder = 'Search industries…';
        } else if (view === 'industry-stocks') {
            searchEl.placeholder = 'Filter stocks…';
        } else if (view === 'scans') {
            searchEl.placeholder = 'Filter stocks…';
        } else if (view === 'watchlists') {
            searchEl.placeholder = 'Search…';
        } else if (view === 'market') {
            searchEl.placeholder = 'Market overview';
        }

        // Show/hide toolbar elements
        document.querySelector('.toolbar').style.display = (view === 'alerts') ? 'none' : '';
        document.getElementById('industry-list-header').style.display = (view === 'industries' && indView === 'list') ? 'flex' : 'none';
        document.getElementById('wl-multichart-toggle-btn').style.display = (view === 'watchlists') ? '' : 'none';

        if (view === 'industries') {
            document.getElementById('view-industries').style.display = 'block';
            renderIndustries();
        } else if (view === 'industry-stocks') {
            document.getElementById('view-industry-stocks').classList.add('active');
        } else if (view === 'scans') {
            document.getElementById('view-scans').style.display = 'flex';
            if (_scanReturnState) {
                var _sr = _scanReturnState;
                _scanReturnState = null;
                if (_sr.ticker) openChartModal(_sr.ticker);
                else if (_sr.snpOpen) snpBuild();
            }
            renderScans();
            scanStartPricePolling();
        } else if (view === 'sector') {
            document.getElementById('view-sector').style.display = 'flex';
        } else if (view === 'watchlists') {
            document.getElementById('view-watchlists').style.display = 'flex';
            wlRender();
            wlStartPricePolling();
            // Auto-load last selected ticker if available
            if (!wlMcActive) {
                var autoTicker = wlChartTicker;
                if (!autoTicker) {
                    var autoList = wlGetLastList();
                    var autoAll  = wlGetAll();
                    if (autoList && autoAll[autoList] && autoAll[autoList].length) {
                        autoTicker = autoAll[autoList][0];
                    }
                }
                if (autoTicker) {
                    wlSelectTicker(autoTicker);
                    var _wlRestoreRows = Array.from(document.querySelectorAll('.wl-ticker-row'));
                    var _wlRestoreIdx  = _wlRestoreRows.findIndex(function(r) { return r.getAttribute('data-wl-ticker') === autoTicker; });
                    if (_wlRestoreIdx >= 0) currentWlIndex = _wlRestoreIdx;
                }
            }
        } else if (view === 'market') {
            document.getElementById('view-market').style.display = 'flex';
            renderMarketBreadth();
            renderMarketHL();
            renderMarketMA();
            renderMarketMovers();
            renderSectorPerf();
            renderIndBreadth();
            renderRSDist();
            marketFetchAll();
            marketFetchMacro();
        } else if (view === 'alerts') {
            document.getElementById('view-alerts').style.display = 'flex';
            alDismissMissed();
            renderAlerts();
            if (!alertPriceTimer && !alertOpenTimer) alStartBackgroundPolling();
        }

        // Restore chart modal + side panel if returning from an alert add
        if (_alReturnState && _alReturnState.view === view) {
            var _rs = _alReturnState;
            _alReturnState = null;
            setTimeout(function() {
                if (_rs.ticker) openChartModal(_rs.ticker);
                if (_rs.snpOpen) snpBuild();
            }, 50);
        }
    };

    // ── Nav tab routing — captures scans state BEFORE modal closes ────────
    window.navTo = function(view) {
        var _modalOpen = document.getElementById('mc-fullscreen-overlay').classList.contains('open');
        var _snpOpen   = document.getElementById('scan-nav-panel').classList.contains('snp-open');
        var _wasScans  = (currentView === 'scans');
        var _wasIndStocks = (currentView === 'industry-stocks');
        if (_modalOpen) { closeChartModal(); }
        // If modal was open while browsing an industry's stock table, just close
        // the modal and stay — don't navigate back to the full industry list.
        if (_modalOpen && _wasIndStocks && view === 'industries') return;
        // Set AFTER close calls so the closeChartModal wrapper doesn't wipe it
        if (_wasScans && view !== 'scans' && (_modalOpen || _snpOpen)) {
            var _sym = _modalOpen ? (document.getElementById('mc-fullscreen-sym').textContent.trim() || null) : null;
            var _sec = (_sym && tickerMap && tickerMap[_sym]) ? (tickerMap[_sym].sector   || '') : '';
            var _ind = (_sym && tickerMap && tickerMap[_sym]) ? (tickerMap[_sym].industry || '') : '';
            _scanReturnState = { ticker: _sym, sector: _sec, industry: _ind, snpOpen: _snpOpen };
        }
        if (view === 'industries') { navToIndustries(); } else { showView(view); }
    };

