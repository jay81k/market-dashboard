
    var multichartActive = false;
    var mcTimeframe      = 'D';
    var mcCols           = parseInt(localStorage.getItem('mcSharedCols') || '4');
    var mcTickers        = [];
    var mcWidgets        = {};

    // Chart watermark font sizes (ticker line / company-name line). Used by the fullscreen chart,
    // the watchlist chart, and the themechange handler that re-applies the watermark, so all three
    // stay in sync. Original was 30 / 14; tried 36 / 17 (too big); now halfway between.
    var _MC_WM_SYM_SIZE  = 33;
    var _MC_WM_NAME_SIZE = 15.5;
    // Weight of the ticker line only (the company name stays regular so long names don't get wider).
    // Passed to the library as the watermark line's fontStyle. Set to '' to go back to regular weight.
    var _MC_WM_SYM_STYLE = 'bold';

    window.setMcCols = function(n) {
        mcCols = n;
        document.querySelectorAll('#stocks-multichart-view .mc-col-btn').forEach(function(b){
            b.classList.toggle('active', +b.getAttribute('data-cols') === n);
        });
        document.getElementById('multichart-grid').style.gridTemplateColumns = 'repeat(' + n + ', 1fr)';
    };

    window.toggleMultichart = function() {
        multichartActive = !multichartActive;
        document.getElementById('stocks-table-view').style.display      = multichartActive ? 'none' : 'flex';
        document.getElementById('stocks-multichart-view').style.display = multichartActive ? 'flex' : 'none';
        document.getElementById('multichart-toggle-btn').style.background = multichartActive ? 'var(--bg-accent-active-2)' : '';
        document.getElementById('multichart-toggle-btn').style.borderColor = multichartActive ? 'var(--accent)' : '';
        document.getElementById('multichart-toggle-btn').style.color = multichartActive ? 'var(--accent-strong)' : '';
        if (multichartActive) renderMulticharts();
    };

    window.setMcTf = function(tf) {
        mcTimeframe = tf;
        document.querySelectorAll('#stocks-multichart-view .mc-tf-btn').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-tf') === tf); });
        renderMulticharts();
    };

    // ── LW Multichart Infrastructure ─────────────────────────────────────────

    var _mcOhlcvCache   = {};   // { "AAPL_D": [...ohlcv] }
    var _mcOhlcvCacheAt = {};   // { "AAPL_D": <Date.now() of last successful fetch> }
    var MC_CACHE_TTL_MS = 30 * 60 * 1000; // how long a cached OHLCV entry is served without a
    // fresh fetch. Bounds exposure to stale data (a split/dividend adjustment or a same-day
    // close revision from Yahoo) to at most this long, instead of either "always refetch on
    // every open" (slow, was the actual bug) or "never refetch once cached" (fast, but wrong
    // forever if a correction lands mid-session). _injectChartLiveBar still separately keeps
    // today's price current in real time regardless of this — this only governs the historical
    // series underneath it.
    var _mcMetaCache    = {};   // { "AAPL": { marketState, preMarketPrice, postMarketPrice, ... } }
    var _mcFetchQueue   = {};   // per-tf: { pending:[], active:0, resolvers:{} }
    var MC_FETCH_LIMIT  = 12; // was 6, originally 50 — pre-warm-all is gone and this cap plus the cooldown below are now the real safety net, so this can sit closer to what's actually visible (~12 tiles) instead of being deliberately conservative

    // _mcFetchQueue is never torn down on its own — it only ever gets added
    // to (via fetchMcOhlcv) or drained. A grid view (Market/Industries/Scans/
    // Watchlists, whichever's currently reusing #multichart-grid) can leave
    // tickers sitting in a *_grid queue's `pending` array — either mid-drain
    // when you navigate away, or pre-queued ahead of scroll position by the
    // grid's IntersectionObserver lookahead. Without this, that backlog just
    // keeps draining in the background into whatever view loads next, firing
    // fetches for tickers that have nothing to do with what's on screen.
    // Call this whenever a grid view is torn down. It only drops PENDING
    // (not yet launched) items — anything already in flight can't be
    // cancelled from here (no AbortController in use) and will just resolve
    // into a widget/DOM that's already gone, which is harmless.
    window.mcClearGridQueue = function() {
        Object.keys(_mcFetchQueue).forEach(function(qKey) {
            if (qKey.slice(-5) === '_grid') {
                _mcFetchQueue[qKey].pending = [];
            }
        });
    };

    // Launch pacing — MC_FETCH_LIMIT caps how many requests can be *open* at
    // once, but a 429 rejection comes back fast, so a freed slot can get
    // refilled almost instantly. That turns "12 concurrent" into a much
    // higher effective launch rate once the proxy starts rejecting. This
    // throttles how often a *new* request is allowed to launch. The clock
    // and 429 cooldown live in yahoo-proxy-pace.js now, shared with
    // market.js/state.js — same upstream proxy, one shared backstop instead
    // of three independent ones that can't see each other's traffic. That
    // file must load before this one.
    var MC_LAUNCH_MIN_SPACING = 150;  // was 100, before that 350 — small bump to make 429 storms slightly less likely; the shared cooldown is still the real backstop

    function _mcFsIsOpen() {
        var overlay = document.getElementById('mc-fullscreen-overlay');
        return !!overlay && overlay.classList.contains('open');
    }

    // Fullscreen state
    var _mcFsOhlcv              = [];
    var _mcFsSym               = null;
    var _mcFsTf                = 'D';
    var _mcFsLastCrosshairPrice = null;
    var _mcFsChart       = null;
    var _mcFsWatermark   = null;
    var _mcFsBuiltSym    = null;   // symbol _mcFsChart is currently built for (distinct from _mcFsSym, which is the symbol last requested)
    var _mcFsBuiltTf     = null;   // timeframe _mcFsChart is currently built for
    var _mcFsCandle      = null;
    var _mcFsVol         = null;
    var _mcFsVolMa       = null;   // 50 SMA on volume
    var _mcFsVolData     = null;   // vol SMA dataset (exposed for vol % label)
    var _mcFsMaSeries    = {};
    var _mcFsMaDataMap   = {};   // { key: Map(time => value) } for O(1) MA proximity lookup
    var _mcFsLastCrosshairTime = null;
    var _mcFsVwapSeries  = [];   // array of { series, anchor, color }
    var _mcFsVwapMode    = false;
    var _mcFsVisibleBars = 65;
    var _AVWAP_COLOR     = '#4caf50';
    var _mcFsActiveMas   = { SMA5: true, EMA8: true, EMA21: true, SMA50: true, SMA150: true, SMA200: true };
    var _mcFsKeyHandler  = null;
    var _mcFsLiveTimer   = null;   // repeating single-ticker fetch while fullscreen stays open

    // ── Candle hover tooltip ──────────────────────────────────────────────────
    var _mcFsTooltipEnabled = false;
    var _wlTooltipEnabled   = false;
    var _mcFsVolSmaMap      = null;
    var _wlVolSmaMap        = null;
    var _lwTooltipDiv       = null;

    // Trendline drawing state
    var _mcFsTrendlineMode          = false;   // tool active?
    var _mcFsTrendlines             = [];      // array of { primitive, p1, p2, leftP, rightP, selected, requestUpdate }
    var _mcFsTrendlineFirst         = null;    // kept for compat (unused in new flow)
    var _mcFsTrendSvgOverlay        = null;    // SVG element overlaid on chart for preview
    var _mcFsTrendSvgLine           = null;    // <line> inside the SVG overlay
    // Alert-backed trendlines keep their `extend` flag (it marks the line as one an alert is evaluated on), but the
    // dashed continuation to the right edge is NOT drawn. The alert itself is evaluated in alerts.js
    // (_alTrendlineEval), independent of this drawing. Set to true to show the continuation again.
    var _TRENDLINE_SHOW_CONTINUATION = false;
    function _TRENDLINE_COLOR() { return themeColor('chart-trendline'); } // theme token (--chart-trendline in styles.css: #c8d0dc dark / #9598a2 light). Re-resolved live on every read, so a theme toggle is picked up. Was 'text-emphasis'.
    var _TRENDLINE_SELECTED_COLOR   = '#f9c74f';
    var _mcFsTrendDraw              = { active: false, startTime: null, startPrice: null };
    var _mcFsTrendContRef           = null;    // reference to chart container div
    var _mcFsTrendMoveBound         = false;   // mousemove attached to document once
    var _mcFsSelectedTrendlineIdx   = -1;      // index in _mcFsTrendlines of selected line, -1 = none
    var _mcFsSelectedVwapIdx        = -1;      // index in _mcFsVwapSeries of selected AVWAP, -1 = none
    var _mcFsTrendDragState         = null;    // { tlIdx, anchorSide:'left'|'right' } during anchor drag

    // Measure tool state (mc-fs)
    var _mcFsMeasureMode       = false;
    var _mcFsMeasureActive     = false;
    var _mcFsMeasurePhase      = 0;      // 0=idle, 1=anchor set (two-click mode preview)
    var _mcFsMeasureRafId      = null;   // rAF throttle handle
    var _mcFsMeasureStart      = null;   // { time, price, barIdx }
    var _mcFsMeasureResult     = null;   // persisted after drag ends
    var _mcFsMeasureList       = [];     // committed measurements — stay until deleted (mutated in place, never reassigned)
    var _mcFsMeasureSvgOverlay = null;
    var _mcFsMeasureSvgRect    = null;
    var _mcFsMeasureHLine      = null;
    var _mcFsMeasureInfoDiv    = null;

    // ── Watchlist LW chart state (mirrors _mcFs* for the side-panel chart) ───
    var _wlOhlcv              = [];
    var _wlSym                = null;
    var _wlTf                 = 'D';
    var _wlLastCrosshairPrice = null;
    var _wlChart              = null;
    var _wlWatermark          = null;
    var _wlCandle             = null;
    var _wlVol                = null;
    var _wlVolMa              = null;
    var _wlVolData            = null;
    var _wlMaSeries           = {};
    var _wlMaDataMap          = {};
    var _wlLastCrosshairTime  = null;
    var _wlVwapSeries         = [];
    var _wlVwapMode           = false;
    var _wlVisibleBars        = 252;
    var _wlActiveMas          = { SMA5: true, EMA8: true, EMA21: true, SMA50: true, SMA150: true, SMA200: true };
    var _wlKeyHandler         = null;
    var _wlTrendlineMode      = false;
    var _wlTrendlines         = [];
    var _wlTrendlineFirst     = null;
    var _wlTrendSvgOverlay    = null;
    var _wlTrendSvgLine       = null;
    var _wlTrendDraw          = { active: false, startTime: null, startPrice: null };
    var _wlTrendContRef       = null;
    var _wlTrendMoveBound     = false;
    var _wlSelectedTrendlineIdx = -1;
    var _wlSelectedVwapIdx      = -1;
    var _wlTrendDragState       = null;
    var _wlCtxPrice             = null;
    var _wlCtxMa                = null;
    var _wlCtxTrendline         = null; // {p1, p2} when right-clicking on a trendline
    var _wlCtxAvwap             = null; // {anchorIdx, anchorTime} when right-click lands on an AVWAP line
    var _wlCtxAttached          = false;

    // Re-theme one multichart grid's cells. Every grid built by _buildLwMcGrid
    // stores its charts in its own registry object (mcWidgets for industries,
    // wlMcWidgets, scansMcWidgets, alMcWidgets), so the theme handler below has
    // to walk each one — walking only mcWidgets left the watchlist, scans and
    // alerts grids with their old-theme canvases after a toggle. Each cell gets
    // its own try/catch so a single failing chart can't skip the ones after it,
    // and failures are logged instead of being swallowed silently.
    function _mcRethemeGridCells(registry, label) {
        if (!registry) return;
        Object.keys(registry).forEach(function(sym) {
            try {
                var inst = registry[sym];
                if (!inst || !inst.chart || !inst.candle) return;
                inst.chart.applyOptions({
                    layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted') },
                    grid:    { vertLines: { color: themeColor('mc-cell-grid') }, horzLines: { color: themeColor('mc-cell-grid') } },
                    rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted') },
                    timeScale: { borderColor: themeColor('bg-surface') },
                });
                inst.candle.applyOptions({
                    upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'),
                    wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
                });
                if (inst.vol && inst.ohlcv && inst.ohlcv.length) {
                    inst.vol.setData(inst.ohlcv.map(function(d) {
                        return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
                    }));
                }
            } catch (e) {
                console.warn('[theme] failed to re-theme ' + label + ' multichart cell ' + sym, e);
            }
        });
    }

    // Re-theme every currently-open chart live if the toggle is flipped —
    // covers every chart shape this file renders: the cells of all four
    // multichart grids (industries, watchlists, scans, alerts — potentially
    // several charts at once), the fullscreen chart, and the watchlist
    // side-panel chart. Anything not currently open just picks up the new
    // theme the next time it's (re)built, same as before.
    window.addEventListener('themechange', function() {
        // wlMcWidgets / scansMcWidgets / alMcWidgets are declared in scripts that
        // load after this one, so guard with typeof rather than assuming they exist.
        _mcRethemeGridCells(mcWidgets, 'industries');
        _mcRethemeGridCells(typeof wlMcWidgets     !== 'undefined' ? wlMcWidgets     : null, 'watchlists');
        _mcRethemeGridCells(typeof scansMcWidgets  !== 'undefined' ? scansMcWidgets  : null, 'scans');
        _mcRethemeGridCells(typeof alMcWidgets     !== 'undefined' ? alMcWidgets     : null, 'alerts');

        [
            { chart: _mcFsChart, candle: _mcFsCandle, vol: _mcFsVol, watermark: _mcFsWatermark, ohlcv: _mcFsOhlcv, sym: _mcFsSym },
            { chart: _wlChart,   candle: _wlCandle,   vol: _wlVol,   watermark: _wlWatermark,   ohlcv: _wlOhlcv,   sym: _wlSym },
        ].forEach(function(c) {
            if (!c.chart || !c.candle) return;
            try {
                c.chart.applyOptions({
                    layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted'), panes: { separatorColor: themeColor('bg-subtle'), separatorHoverColor: themeColor('bg-surface-alpha') } },
                    rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted') },
                    timeScale: { borderColor: themeColor('bg-surface') },
                });
                c.candle.applyOptions({
                    upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'),
                    wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
                });
                if (c.vol) {
                    c.vol.applyOptions({ color: themeColor('al-chart-volume') });
                    c.vol.priceScale().applyOptions({ borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted') });
                    if (c.ohlcv && c.ohlcv.length) {
                        c.vol.setData(c.ohlcv.map(function(d) {
                            return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
                        }));
                    }
                }
                if (c.watermark && c.sym) {
                    var _wmMeta        = _mcMetaCache[c.sym] || {};
                    var _wmCompanyName = _wmMeta.longName || _wmMeta.shortName || '';
                    c.watermark.applyOptions({
                        lines: [
                            { text: c.sym, color: themeColor('chart-watermark'), fontSize: _MC_WM_SYM_SIZE, fontStyle: _MC_WM_SYM_STYLE },
                            _wmCompanyName ? { text: _wmCompanyName, color: themeColor('chart-watermark'), fontSize: _MC_WM_NAME_SIZE } : null,
                        ].filter(Boolean),
                    });
                }
            } catch (e) {}
        });
        // Canvas-drawn trendlines repaint on the applyOptions calls above because
        // _TRENDLINE_COLOR() is re-read on every draw. The SVG preview lines do NOT:
        // their stroke is a static attribute. They stay hidden between uses and have
        // their stroke re-resolved each time they are shown (see _onTrendMouseDownCore),
        // so nothing further is needed here.
    });

    // Measure tool state (wl)
    var _wlMeasureMode       = false;
    var _wlMeasureActive     = false;
    var _wlMeasurePhase      = 0;
    var _wlMeasureRafId      = null;
    var _wlMeasureStart      = null;
    var _wlMeasureResult     = null;
    var _wlMeasureList       = [];       // committed measurements — stay until deleted (mutated in place, never reassigned)
    var _wlMeasureSvgOverlay = null;
    var _wlMeasureSvgRect    = null;
    var _wlMeasureHLine      = null;
    var _wlMeasureInfoDiv    = null;

    // Render-token per multichart context (prevents stale renders)
    var _mcRenderTokens = { ind: 0, wl: 0, scans: 0, al: 0 };

    // MA definitions (period, color, type)
    var _MC_MA_DEFS = {
        SMA5:   { period: 5,   color: '#673ab7', ema: false },
        EMA8:   { period: 8,   color: '#f48fb1', ema: true  },
        EMA21:  { period: 21,  color: '#1848cc', ema: true  },
        SMA50:  { period: 50,  color: '#f23645', ema: false },
        SMA150: { period: 150, color: '#757575', ema: false },
        // SMA200 color is read at chart-build time from --al-chart-ma200 so it can
        // differ per theme (pure black in light mode); falls back to the old value.
        SMA200: { period: 200, get color() { return (typeof themeColor === 'function' && themeColor('al-chart-ma200')) || '#2e2e2e'; }, ema: false },
    };

    function _mcInterval(tf) { return tf === 'W' ? '1wk' : tf === 'M' ? '1mo' : '1d'; }
    function _mcRange(tf, gridMode) {
        // Grid tiles are small — 10y of daily bars is far more than a ~150px-wide
        // cell can show, and it's the main thing making the first load slow. 2y
        // still comfortably covers the longest MA shown on tiles (SMA200 needs
        // 200 trading days ≈ 10 months; 2y leaves ~14 months of margin beyond
        // that for the line to actually be visible, not just barely present).
        // Fullscreen/watchlist views (gridMode omitted) are unchanged — full
        // history, exactly as before.
        if (gridMode && tf === 'D') return '2y';
// NOTE: Yahoo silently coerces interval=1wk/1mo + range=max into 3-month
// (quarterly) bars — the response comes back stamped dataGranularity:'3mo'
// no matter what interval was requested, which is why Weekly and Monthly
// rendered identically. Use explicit ranges Yahoo actually honors:
// 10y of weekly bars (~523, covers SMA200) and 20y of monthly bars
// (~241, covers SMA200). Verified against the live proxy 2026-09-20.
if (tf === 'W') return '10y';
if (tf === 'M') return '20y';
return '10y';

    }

    // ── Weekly / Monthly in-progress bar ─────────────────────────────────────
    // Bars are stamped noon-UTC by fetchMcOhlcv. Weekly bars span Mon-Sun and
    // monthly bars a calendar month. _mcPeriodKey collapses any such stamp to an
    // integer naming its bar's period, so "is this the current bar?" is a key
    // comparison that never depends on which day Yahoo stamps a period with
    // (Monday vs. first session, the 1st vs. first trading day).
    function _mcPeriodKey(ts, tf) {
        var day = Math.floor(ts / 86400);
        if (tf === 'W') return Math.floor((day + 3) / 7); // epoch day 0 is a Thursday -> Monday-based week index
        if (tf === 'M') { var d = new Date(ts * 1000); return d.getUTCFullYear() * 12 + d.getUTCMonth(); }
        return day;
    }
    function _mcPeriodStartTs(ts, tf) {
        var d = new Date(ts * 1000);
        if (tf === 'W') return ts - ((d.getUTCDay() + 6) % 7) * 86400;
        if (tf === 'M') return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1) / 1000 + 43200;
        return ts;
    }
    // Folds a live price into the current W/M bar (mutating ohlcvArr's last bar,
    // same as the Daily paths do) and returns a bar for series.update(), or null
    // if nothing should be drawn. If the feed has no bar for the current period
    // yet (first session of a new week/month), a new one is created only when
    // createIfMissing is set AND the market is open - never a phantom after close.
    function _mcApplyLiveWM(ohlcvArr, tf, price, dayHigh, dayLow, createIfMissing) {
        if (!ohlcvArr || !ohlcvArr.length || !price) return null;
        var now = new Date();
        var todayTs = Math.floor(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 1000) + 43200;
        var last = ohlcvArr[ohlcvArr.length - 1];
        if (_mcPeriodKey(last.time, tf) === _mcPeriodKey(todayTs, tf)) {
            last.high  = dayHigh != null ? Math.max(last.high, dayHigh, price) : Math.max(last.high, price);
            last.low   = dayLow  != null ? Math.min(last.low,  dayLow,  price) : Math.min(last.low,  price);
            last.close = price;
            return { time: last.time, open: last.open, high: last.high, low: last.low, close: last.close, volume: last.volume };
        }
        if (!createIfMissing || !wlIsMarketOpen()) return null;
        var t = _mcPeriodStartTs(todayTs, tf);
        ohlcvArr.push({ time: t, open: price, high: price, low: price, close: price, volume: 0 });
        return { time: t, open: price, high: price, low: price, close: price, volume: 0 };
    }
    // Yahoo can return the latest trading day as its own row alongside the
    // week/month row, so one period arrives as two candles. Collapse rows that
    // share a period into one bar: open from the earliest row, high/low across
    // both, close from the latest. Volume takes the larger of the two - if the
    // extra row is day-only that understates the period, if it is cumulative
    // it is exact; it can never double-count. No-op when every period has one row.
    function _mcMergeSamePeriod(ohlcv, tf) {
        if (tf === 'D' || !ohlcv || ohlcv.length < 2) return ohlcv;
        var out = [];
        for (var i = 0; i < ohlcv.length; i++) {
            var b = ohlcv[i], p = out[out.length - 1];
            if (p && _mcPeriodKey(p.time, tf) === _mcPeriodKey(b.time, tf)) {
                if (b.high != null) p.high = Math.max(p.high, b.high);
                if (b.low  != null) p.low  = Math.min(p.low,  b.low);
                p.close  = b.close;
                p.volume = Math.max(p.volume || 0, b.volume || 0);
            } else {
                out.push(b);
            }
        }
        return out;
    }

    // Concurrent fetch queue — max MC_FETCH_LIMIT in-flight at once.
    //
    // force === true refetches even if the cached entry is still inside the TTL. The OLD entry
    // stays in _mcOhlcvCache until the new one arrives, and stays if the refetch fails (a forced
    // call then resolves null so the caller knows). Callers must never `delete` a good series
    // just to refresh it -- that throws away history on every transient failure.
    //
    // Without force, an entry older than MC_CACHE_TTL_MS is refetched, but if that refetch fails
    // the stale entry is served instead of null (stale data beats a blank chart).
    function fetchMcOhlcv(sym, tf, gridMode, force) {
        var key = sym + '_' + tf + (gridMode ? '_grid' : '');
        var isFresh = !force && _mcOhlcvCache[key] !== undefined &&
            (Date.now() - (_mcOhlcvCacheAt[key] || 0)) < MC_CACHE_TTL_MS;
        if (isFresh) return Promise.resolve(_mcOhlcvCache[key]);
        var pr = new Promise(function(resolve) {
            var qKey = tf + (gridMode ? '_grid' : '');
            if (!_mcFetchQueue[qKey]) _mcFetchQueue[qKey] = { pending: [], active: 0, resolvers: {}, forced: {} };
            var q = _mcFetchQueue[qKey];
            if (!q.forced) q.forced = {};
            if (!q.resolvers[sym]) q.resolvers[sym] = [];
            q.resolvers[sym].push(resolve);
            if (force) q.forced[sym] = true;
            if (q.pending.indexOf(sym) === -1) q.pending.push(sym);
            _drainMcQueue(tf, gridMode);
        });
        if (force) return pr;
        return pr.then(function(r) {
            return (r === null && _mcOhlcvCache[key] !== undefined) ? _mcOhlcvCache[key] : r;
        });
    }

    function _drainMcQueue(tf, gridMode) {
        var qKey = tf + (gridMode ? '_grid' : '');
        var q = _mcFetchQueue[qKey];
        if (!q) return;

        // Grid tiles stand down entirely while fullscreen is open. The gate in
        // attemptLoad() only stops *new* attemptLoad calls from adding to this
        // queue — it does nothing about requests already sitting in q.pending
        // from before the overlay opened, which would otherwise keep draining
        // on their own timers, keep resetting the shared launch clock, and
        // could still trip the shared 429 cooldown that then blocks the
        // fullscreen chart's own request. This closes that gap.
        if (gridMode && _mcFsIsOpen()) {
            if (!q._fsGateTimer) {
                q._fsGateTimer = setTimeout(function() {
                    q._fsGateTimer = null;
                    _drainMcQueue(tf, gridMode);
                }, 1000); // same poll cadence as the attemptLoad() gate
            }
            return;
        }

        if (Date.now() < window.yahooProxyPace.cooldownUntil()) {
            if (!q._resumeTimer) {
                var waitMs = window.yahooProxyPace.cooldownUntil() - Date.now() + 25;
                q._resumeTimer = setTimeout(function() {
                    q._resumeTimer = null;
                    _drainMcQueue(tf, gridMode);
                }, waitMs);
            }
            return;
        }

        while (q.pending.length > 0 && q.active < MC_FETCH_LIMIT) {
            var sym = q.pending[0];
            var key = sym + '_' + tf + (gridMode ? '_grid' : '');
            var _entryFresh = _mcOhlcvCache[key] !== undefined &&
                (Date.now() - (_mcOhlcvCacheAt[key] || 0)) < MC_CACHE_TTL_MS;
            if (_entryFresh && !(q.forced && q.forced[sym])) {
                q.pending.shift();
                var res0 = (q.resolvers[sym] || []).splice(0);
                delete q.resolvers[sym];
                res0.forEach(function(r) { r(_mcOhlcvCache[key]); });
                continue;
            }

            var sinceLast = Date.now() - window.yahooProxyPace.lastLaunchAt();
            if (sinceLast < MC_LAUNCH_MIN_SPACING) {
                if (!q._paceTimer) {
                    var _spaceWait = MC_LAUNCH_MIN_SPACING - sinceLast;
                    q._paceTimer = setTimeout(function() {
                        q._paceTimer = null;
                        _drainMcQueue(tf, gridMode);
                    }, _spaceWait);
                }
                break;
            }

            var _retryAttempt = (q.retryAttempt && q.retryAttempt[sym]) || 0;
            if (q.retryAttempt) delete q.retryAttempt[sym];
            q.pending.shift();
            q.active++;
            (function doFetch(s, attempt) {
                window.yahooProxyPace.markLaunched();
                var url = WL_PROXY + '?symbol=' + encodeURIComponent(s) + '&interval=' + _mcInterval(tf) + '&range=' + _mcRange(tf, gridMode);
                fetch(url).then(function(resp) {
                        if (resp.ok) return resp.json();
                        // Drain the body before treating this as an error —
                        // an unread body on a non-ok response is what makes
                        // Cloudflare Workers cancel stalled in-flight requests.
                        var retryAfter = resp.headers.get('Retry-After');
                        if (resp.status === 429) window.yahooProxyPace.register429();
                        return resp.text().catch(function() {}).then(function() {
                            var err = new Error('http_' + resp.status);
                            err.status = resp.status;
                            err.retryAfter = retryAfter;
                            throw err;
                        });
                    })
                    .then(function(data) {
                        var result = data && data.chart && data.chart.result && data.chart.result[0];
                        if (result && result.meta) _mcMetaCache[s] = result.meta;
                        var ohlcv = [];
                        if (result && result.timestamp) {
                            var ts = result.timestamp;
                            var qt = result.indicators && result.indicators.quote && result.indicators.quote[0];
                            if (qt) {
                                for (var i = 0; i < ts.length; i++) {
                                    if (!ts[i] || qt.open[i] == null || qt.close[i] == null) continue;
                                    // Normalize to noon UTC (12:00 UTC) so LWC renders the correct
                                    // calendar date in any local timezone. Yahoo historical bars use
                                    // midnight UTC; noon UTC is safely within the correct day for
                                    // all US/EU/Asia markets and avoids the -1 day shift for UTC-N zones.
                                    var _noonTs = Math.floor(ts[i] / 86400) * 86400 + 43200;
                                    ohlcv.push({ time: _noonTs, open: qt.open[i], high: qt.high[i], low: qt.low[i], close: qt.close[i], volume: qt.volume[i] || 0 });
                                }
                                // LWC requires strictly monotonic timestamps — dedupe and sort defensively
                                var _tsSeen = {};
                                ohlcv = ohlcv.filter(function(d) { if (_tsSeen[d.time]) return false; _tsSeen[d.time] = true; return true; });
                                ohlcv.sort(function(a, b) { return a.time - b.time; });
                            }
                        }
                        // The in-progress week/month bar is deliberately KEPT (TradingView draws
                        // it too). It is kept current by _mcApplyLiveWM via _injectChartLiveBar,
                        // _mcFsStartLiveTick, _alStartLiveTick and _updateMcLiveCandle.
                        ohlcv = _mcMergeSamePeriod(ohlcv, tf); // one candle per week/month even if Yahoo splits the period
                        // An empty answer for a ticker we already hold history for is a failed refresh, not "no data":
                        // throw so it is retried, and the existing series stays in place if it keeps failing.
                        var _prevSeries = _mcOhlcvCache[s + '_' + tf + (gridMode ? '_grid' : '')];
                        if (ohlcv.length === 0 && _prevSeries && _prevSeries.length > 0) throw new Error('empty_refresh');
                        _mcOhlcvCache[s + '_' + tf + (gridMode ? '_grid' : '')] = ohlcv;
                        _mcOhlcvCacheAt[s + '_' + tf + (gridMode ? '_grid' : '')] = Date.now();
                        if (q.forced) delete q.forced[s];
                        var res = (q.resolvers[s] || []).splice(0);
                        delete q.resolvers[s];
                        res.forEach(function(r) { r(ohlcv); });
                        q.active = Math.max(0, q.active - 1);
                        _drainMcQueue(tf, gridMode);
                    })
                    .catch(function(err) {
                        var MAX_RETRIES = 3;
                        if (attempt < MAX_RETRIES) {
                            var delayMs;
                            if (err && err.retryAfter && !isNaN(parseInt(err.retryAfter, 10))) {
                                delayMs = parseInt(err.retryAfter, 10) * 1000;
                            } else {
                                // Exponential backoff: 1s, 2s, 4s
                                delayMs = 1000 * Math.pow(2, attempt);
                            }
                            delayMs = Math.min(delayMs, 10000); // don't let one stubborn ticker hold a slot forever
                            // q.active stays held during the backoff so the slot isn't reused.
                            // IMPORTANT: don't call doFetch directly here. Multiple symbols
                            // retrying at once would each independently decide "the cooldown's
                            // over" and fire in the same instant, re-tripping it together as a
                            // group (a thundering herd) — that's what the checked-cooldownUntil
                            // version of this still did. Routing back through _drainMcQueue means
                            // every relaunch, retry or fresh, goes through the one gate that
                            // already serializes on both cooldown and launch spacing, same as
                            // every other consumer in the app.
                            setTimeout(function() {
                                q.retryAttempt = q.retryAttempt || {};
                                q.retryAttempt[s] = attempt + 1;
                                q.pending.unshift(s);
                                q.active = Math.max(0, q.active - 1);
                                _drainMcQueue(tf, gridMode);
                            }, delayMs);
                        } else {
                            // Don't cache this as confirmed-empty data — it's a failure, not
                            // "no data for this ticker." Leaving the cache key unset means a
                            // later retry (e.g. clicking the failed tile) triggers a real
                            // fetch instead of instantly resolving to a stale permanent blank.
                            var res = (q.resolvers[s] || []).splice(0);
                            delete q.resolvers[s];
                            if (q.forced) delete q.forced[s];
                            res.forEach(function(r) { r(null); }); // null = failed, distinct from [] = genuinely no data
                            q.active = Math.max(0, q.active - 1);
                            _drainMcQueue(tf, gridMode);
                        }
                    });
            })(sym, _retryAttempt);
        }
    }

    // ── MA / AVWAP maths ──────────────────────────────────────────────────────
    function _calcSMA(ohlcv, period) {
        var out = [];
        for (var i = period - 1; i < ohlcv.length; i++) {
            var sum = 0;
            for (var j = i - (period - 1); j <= i; j++) sum += ohlcv[j].close;
            out.push({ time: ohlcv[i].time, value: sum / period });
        }
        return out;
    }
    function _calcEMA(ohlcv, period) {
        var out = [], k = 2 / (period + 1), ema = ohlcv[0] ? ohlcv[0].close : 0;
        for (var i = 0; i < ohlcv.length; i++) {
            ema = ohlcv[i].close * k + ema * (1 - k);
            if (i >= period - 1) out.push({ time: ohlcv[i].time, value: ema });
        }
        return out;
    }
    // Price input for AVWAP. 'hlc3' = (H+L+C)/3, the conventional VWAP "typical price" (what TradingView's
    // anchored VWAP defaults to). 'ohlc4' was the previous behaviour -- the open is not a traded-volume
    // price, so it skews the line toward gap opens. Flip this one constant to go back.
    var AVWAP_PRICE_SOURCE = 'hlc3';
    function _calcAVWAP(ohlcv, anchorIdx) {
        var out = [], cumVT = 0, cumV = 0;
        for (var i = anchorIdx; i < ohlcv.length; i++) {
            var tp = (AVWAP_PRICE_SOURCE === 'ohlc4')
                ? (ohlcv[i].open + ohlcv[i].high + ohlcv[i].low + ohlcv[i].close) / 4
                : (ohlcv[i].high + ohlcv[i].low + ohlcv[i].close) / 3;
            cumVT += tp * (ohlcv[i].volume || 0);
            cumV  += (ohlcv[i].volume || 0);
            if (cumV > 0) out.push({ time: ohlcv[i].time, value: cumVT / cumV });
        }
        return out;
    }
    function _calcMA(ohlcv, key) {
        var def = _MC_MA_DEFS[key];
        if (!def) return [];
        return def.ema ? _calcEMA(ohlcv, def.period) : _calcSMA(ohlcv, def.period);
    }
    function _maLabel(key) {
        var def = _MC_MA_DEFS[key];
        if (!def) return key;
        return (def.ema ? 'EMA' : 'SMA') + ' ' + def.period;
    }
    function _barIdxByTime(ohlcv, time) {
        for (var i = 0; i < ohlcv.length; i++) { if (ohlcv[i].time >= time) return i; }
        return -1;
    }

    // ── Candle hover tooltip helpers ──────────────────────────────────────────
    function _getLwTooltipDiv() {
        if (!_lwTooltipDiv) {
            _lwTooltipDiv = document.createElement('div');
            _lwTooltipDiv.id = 'lw-hover-tooltip';
            _lwTooltipDiv.style.cssText = 'position:fixed;z-index:9999;pointer-events:none;display:none;' +
                'background:var(--bg-surface);border:1px solid var(--border);border-radius:5px;' +
                'padding:8px 12px;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums;' +
                'font-family:inherit;color:var(--text-primary-alt);line-height:1.75;white-space:nowrap;' +
                'box-shadow:0 4px 20px rgba(0,0,0,0.6);';
            document.body.appendChild(_lwTooltipDiv);
        }
        return _lwTooltipDiv;
    }
    function _positionTooltip(div, cx, cy, rightBound) {
        var W = window.innerWidth, H = window.innerHeight;
        var tw = div.offsetWidth  || 180;
        var th = div.offsetHeight || 240;
        var rb = (rightBound != null ? rightBound : W) - 8;
        var x = cx + 18, y = cy + 18;
        if (x + tw > rb) x = cx - tw - 18;
        if (y + th > H - 8) y = cy - th - 18;
        div.style.left = Math.max(8, x) + 'px';
        div.style.top  = Math.max(8, y) + 'px';
    }
    function _fmtBarDate(time) {
        var d = new Date(time * 1000);
        return (d.getUTCMonth() + 1) + '/' + d.getUTCDate() + '/' + d.getUTCFullYear();
    }
    function _buildTooltipHtml(d, barIdx, ohlcv, volSmaMap, maDataMap, activeMas, barTime) {
        function fp(v) { return v != null ? v.toFixed(2) : '\u2014'; }
        function fv(v) { return v == null ? '\u2014' : v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'K' : v.toFixed(0); }
        var cl     = d.close >= d.open ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
        var delta  = 0, pct = 0, chgClr = 'var(--text-muted)';
        if (barIdx > 0) {
            var prevClose = ohlcv[barIdx - 1].close;
            delta  = d.close - prevClose;
            pct    = (delta / prevClose) * 100;
            chgClr = delta >= 0 ? 'var(--success)' : 'var(--danger)';
        }
        var cr    = (d.high > d.low) ? Math.round((d.close - d.low) / (d.high - d.low) * 100) : null;
        var crClr = cr != null ? (cr >= 60 ? 'var(--success)' : cr >= 30 ? 'var(--warning-alt)' : 'var(--danger)') : 'var(--text-muted)';
        var vol   = ohlcv[barIdx] ? ohlcv[barIdx].volume : null;
        var L = '<span style="color:var(--text-muted)">', V = '<span style="color:var(--text-primary-alt)">', E = '</span>';
        var html = '<div style="color:var(--text-muted-2);margin-bottom:3px;">' + _fmtBarDate(barTime) + '</div>';
        var lastVal = '<span>' + V + fp(d.close) + E;
        if (barIdx > 0) lastVal += ' <span style="color:' + chgClr + '">' + (delta >= 0 ? '+$' : '-$') + Math.abs(delta).toFixed(2) + E;
        lastVal += '</span>';
        var ohlcvRows =
            L + 'Open'  + E + V + fp(d.open)  + E +
            L + 'High'  + E + V + fp(d.high)  + E +
            L + 'Low'   + E + V + fp(d.low)   + E +
            L + 'Last'  + E + lastVal +
            L + '% Chg' + E + '<span style="color:' + chgClr + '">' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' + E +
            L + 'CR%'   + E + (cr != null ? '<span style="color:' + crClr + '">' + cr + '%' + E : L + '\u2014' + E) +
            L + 'Vol'   + E + V + fv(vol) + E;
        if (vol != null && volSmaMap) {
            var smaVal = volSmaMap.get(barTime);
            if (smaVal && smaVal > 0) {
                var vp    = (vol / smaVal - 1) * 100;
                var vpClr = vp >= 0 ? 'var(--success)' : 'var(--danger)';
                ohlcvRows += L + 'Vol % Chg' + E + '<span style="color:' + vpClr + '">' + (vp >= 0 ? '+' : '') + vp.toFixed(2) + '%' + E;
            }
        }
        html += '<div style="display:grid;grid-template-columns:auto auto;column-gap:10px;row-gap:0;">' + ohlcvRows + '</div>';
        var maOrder = ['SMA5', 'EMA8', 'EMA21', 'SMA50', 'SMA150', 'SMA200'];
        var maRows = [];
        maOrder.forEach(function(key) {
            if (!activeMas[key] || !maDataMap[key]) return;
            var maVal = maDataMap[key].get(barTime);
            if (maVal == null) return;
            var def   = _MC_MA_DEFS[key]; if (!def) return;
            var label = (def.ema ? 'EMA' : 'SMA') + '(' + def.period + ')';
            var dp    = (d.close - maVal) / maVal * 100;
            var dpClr = dp >= 0 ? 'var(--success)' : 'var(--danger)';
            maRows.push(
                '<span style="color:' + def.color + '">' + label + '</span>' +
                '<span style="color:var(--text-primary-alt);justify-self:end">' + fp(maVal) + '</span>' +
                '<span style="color:' + dpClr + ';justify-self:end">' + (dp >= 0 ? '+' : '') + dp.toFixed(1) + '%</span>'
            );
        });
        if (maRows.length) {
            html += '<div style="border-top:1px solid var(--border);margin:5px 0 4px;"></div>';
            html += '<div style="display:grid;grid-template-columns:auto auto auto;column-gap:10px;row-gap:2px;">' + maRows.join('') + '</div>';
        }
        return html;
    }

    // ── Shared measure tool helpers ───────────────────────────────────────────
    function _ensureMeasureOverlay(container, svgClass, infoClass) {
        var svg = container.querySelector('.' + svgClass);
        var rect, hLine;
        if (!svg) {
            svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('class', svgClass);
            svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:6;display:none;';
            rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            svg.appendChild(rect);
            hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            svg.appendChild(hLine);
            container.appendChild(svg);
        } else {
            rect  = svg.querySelector('rect');
            hLine = svg.querySelector('line');
        }
        var info = container.querySelector('.' + infoClass);
        if (!info) {
            info = document.createElement('div');
            info.setAttribute('class', infoClass);
            info.style.cssText = 'position:absolute;display:none;z-index:7;pointer-events:none;color:var(--text-on-accent);font-size:11.5px;font-weight:600;font-family:inherit;font-variant-numeric:tabular-nums;line-height:1.55;padding:5px 9px;border-radius:3px;white-space:nowrap;';
            container.appendChild(info);
        }
        return { svg: svg, rect: rect, hLine: hLine, info: info };
    }

    function _renderMeasureOverlay(chart, candle, contRef, svgEl, rectEl, hLineEl, infoEl, result) {
        if (!result || !chart || !candle || !contRef) return;
        var x1 = chart.timeScale().logicalToCoordinate(result.startBarIdx);
        var x2 = chart.timeScale().logicalToCoordinate(result.endBarIdx);
        var y1 = candle.priceToCoordinate(result.startPrice);
        var y2 = candle.priceToCoordinate(result.endPrice);
        if (x1 == null || y1 == null || y2 == null) return;

        var left   = Math.min(x1, x2);
        var right  = Math.max(x1, x2);
        var top    = Math.min(y1, y2);
        var bottom = Math.max(y1, y2);
        var w = right - left;
        var h = bottom - top;
        var isUp = result.endPrice >= result.startPrice;

        var fillClr   = isUp ? 'var(--al-measure-fill-up)'   : 'var(--al-measure-fill-down)';
        var strokeClr = isUp ? 'var(--al-measure-stroke-up)' : 'var(--al-measure-stroke-down)';
        var infoBg    = isUp ? 'var(--al-measure-bg-up)'     : 'var(--al-measure-bg-down)';

        rectEl.setAttribute('x', left);
        rectEl.setAttribute('y', top);
        rectEl.setAttribute('width',  Math.max(w, 1));
        rectEl.setAttribute('height', Math.max(h, 1));
        rectEl.setAttribute('fill',   fillClr);
        rectEl.setAttribute('stroke', strokeClr);
        rectEl.setAttribute('stroke-width', '1');

        var midY = (y1 + y2) / 2;
        hLineEl.setAttribute('x1', left);  hLineEl.setAttribute('y1', midY);
        hLineEl.setAttribute('x2', right); hLineEl.setAttribute('y2', midY);
        hLineEl.setAttribute('stroke', 'var(--al-watermark)');
        hLineEl.setAttribute('stroke-width', '0.5');
        hLineEl.setAttribute('stroke-dasharray', '3,3');

        var dStr = (result.priceDelta >= 0 ? '+' : '') + result.priceDelta.toFixed(2);
        var pStr = (result.pctDelta   >= 0 ? '+' : '') + result.pctDelta.toFixed(2)   + '%';
        infoEl.innerHTML = '<div>' + dStr + ' (' + pStr + ')</div><div>' + _measureLenLabel(result) + '</div>';
        infoEl.style.background = infoBg;
        infoEl.style.display    = '';

        var cRect = contRef.getBoundingClientRect();
        var cw = cRect.width, ch = cRect.height;
        // Cache info-box dimensions after first paint to avoid repeated reflow
        if (!infoEl._cachedW) { infoEl._cachedW = infoEl.offsetWidth  || 115; }
        if (!infoEl._cachedH) { infoEl._cachedH = infoEl.offsetHeight ||  40; }
        var iw = infoEl._cachedW;
        var ih = infoEl._cachedH;
        var gap = 5;
        var iLeft = right  + gap;
        var iTop  = bottom + gap;
        if (iLeft + iw > cw) iLeft = left - iw - gap;
        if (iTop  + ih > ch) iTop  = top  - ih - gap;
        if (iLeft < 0) iLeft = gap;
        if (iTop  < 0) iTop  = gap;
        infoEl.style.left = iLeft + 'px';
        infoEl.style.top  = iTop  + 'px';

        svgEl.style.display = '';
    }

    function _hideMeasureOverlay(svgEl, infoEl) {
        if (svgEl)  svgEl.style.display  = 'none';
        if (infoEl) infoEl.style.display = 'none';
    }

    // ── Committed (persistent) measurements ──────────────────────────────────
    // Finishing a measurement (second click) turns it into an item in a per-chart list. It is drawn as two
    // horizontal lines at the start and end price, a dashed arrow between them pointing at the end price,
    // the duration in days above and the price change below (no box). Items are anchored by time + price, so
    // they stay put through pan/zoom and new bars, and live until deleted (click a line or label to select it,
    // then Delete). Lists are only ever mutated in place. Lifetime matches the chart's trendlines.
    function _measureSavedLayer(container) {
        var layer = container.querySelector('svg.measure-saved-layer');
        if (!layer) {
            layer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            layer.setAttribute('class', 'measure-saved-layer');
            layer.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:6;';
            container.appendChild(layer);
        }
        return layer;
    }

    function _measureMakeEl(layer, tag, attrs) {
        var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (var k in attrs) el.setAttribute(k, attrs[k]);
        layer.appendChild(el);
        return el;
    }

    function _measureEls(m) { return [m.topLine, m.botLine, m.vLine, m.head, m.tTop, m.tBot, m.hStart, m.hEnd]; }

    function _measureCommit(cfg, startTime, startPrice, endTime, endPrice) {
        var layer = _measureSavedLayer(cfg.contRef);
        var txt = { 'font-size': '11.5', 'font-weight': '600', 'text-anchor': 'middle', 'fill': 'var(--text-primary)' };
        var item = { startTime: startTime, startPrice: startPrice, endTime: endTime, endPrice: endPrice, sel: false, box: null };
        item.topLine = _measureMakeEl(layer, 'line', {});
        item.botLine = _measureMakeEl(layer, 'line', {});
        item.vLine   = _measureMakeEl(layer, 'line', { 'stroke-dasharray': '3,3' });
        item.vLine.style.strokeOpacity = 'var(--al-measure-dash-opacity, 1)';   // fainter than the lines; tune in styles.css
        item.head    = _measureMakeEl(layer, 'polygon', {});
        item.tTop    = _measureMakeEl(layer, 'text', txt);
        item.tBot    = _measureMakeEl(layer, 'text', txt);
        // End handles: shown only while the measurement is selected, drag one to adjust that end (see _measureBeginDrag)
        item.hStart  = _measureMakeEl(layer, 'circle', { 'r': '5', 'stroke-width': '2', 'fill': 'var(--bg-surface)' });
        item.hEnd    = _measureMakeEl(layer, 'circle', { 'r': '5', 'stroke-width': '2', 'fill': 'var(--bg-surface)' });
        item.hStart.style.display = 'none';
        item.hEnd.style.display   = 'none';
        cfg.measureList.push(item);
        return item;
    }

    function _measureStyleSel(m) {
        var big = !!m.sel;
        m.topLine.setAttribute('stroke-width', big ? '2.5' : '1.5');
        m.botLine.setAttribute('stroke-width', big ? '2.5' : '1.5');
        m.vLine.setAttribute('stroke-width',   big ? '1.5' : '1');
        if (m.hStart) m.hStart.style.display = big ? '' : 'none';
        if (m.hEnd)   m.hEnd.style.display   = big ? '' : 'none';
    }

    // Draw one committed measurement. Returns the drawn geometry (used for hit-testing), or null if off-scale.
    function _measureRenderItem(chart, candle, contRef, m, res) {
        var ts = chart.timeScale();
        var x1 = ts.logicalToCoordinate(res.startBarIdx), x2 = ts.logicalToCoordinate(res.endBarIdx);
        var y1 = candle.priceToCoordinate(res.startPrice), y2 = candle.priceToCoordinate(res.endPrice);
        if (x1 == null || x2 == null || y1 == null || y2 == null) return null;

        var left = Math.min(x1, x2), right = Math.max(x1, x2), midX = (left + right) / 2;
        var yTop = Math.min(y1, y2), yBot = Math.max(y1, y2);
        var clr  = res.endPrice >= res.startPrice ? 'var(--al-measure-stroke-up)' : 'var(--al-measure-stroke-down)';

        m.topLine.setAttribute('x1', left); m.topLine.setAttribute('x2', right);
        m.topLine.setAttribute('y1', yTop); m.topLine.setAttribute('y2', yTop);
        m.botLine.setAttribute('x1', left); m.botLine.setAttribute('x2', right);
        m.botLine.setAttribute('y1', yBot); m.botLine.setAttribute('y2', yBot);
        m.topLine.setAttribute('stroke', clr);
        m.botLine.setAttribute('stroke', clr);

        // Dashed arrow from the start-price line to the end-price line, head on the end-price line
        var dir  = y2 >= y1 ? 1 : -1;                  // +1 = pointing down the screen
        var hLen = Math.min(13, Math.abs(y2 - y1));    // head length, never longer than the gap
        m.vLine.setAttribute('x1', midX); m.vLine.setAttribute('x2', midX);
        m.vLine.setAttribute('y1', y1);   m.vLine.setAttribute('y2', y2 - dir * hLen);
        m.vLine.setAttribute('stroke', clr);
        m.head.setAttribute('points', midX + ',' + y2 + ' ' + (midX - 6) + ',' + (y2 - dir * hLen) + ' ' + (midX + 6) + ',' + (y2 - dir * hLen));
        m.head.setAttribute('fill', clr);

        // End handles sit exactly on the two anchors: start = (start candle, start price), end = (end candle, end price)
        m.hStart.setAttribute('cx', x1); m.hStart.setAttribute('cy', y1); m.hStart.setAttribute('stroke', clr);
        m.hEnd.setAttribute('cx',   x2); m.hEnd.setAttribute('cy',   y2); m.hEnd.setAttribute('stroke',   clr);

        // Labels: duration above the top line, change below the bottom line (flipped inside if off the edge)
        var ch = contRef.getBoundingClientRect().height;
        var dStr = (res.priceDelta >= 0 ? '+' : '') + res.priceDelta.toFixed(2);
        var pStr = (res.pctDelta   >= 0 ? '+' : '') + res.pctDelta.toFixed(2) + '%';
        m.tTop.textContent = _measureLenLabel(res);
        m.tTop.setAttribute('x', midX);
        m.tTop.setAttribute('y', (yTop - 8 >= 12) ? yTop - 8 : yTop + 16);
        m.tBot.textContent = dStr + ' (' + pStr + ')';
        m.tBot.setAttribute('x', midX);
        m.tBot.setAttribute('y', (yBot + 16 <= ch - 4) ? yBot + 16 : yBot - 8);

        _measureStyleSel(m);
        return { left: left, right: right, top: yTop, bottom: yBot, midX: midX, x1: x1, y1: y1, x2: x2, y2: y2 };
    }

    function _measureRenderAll(list, chart, candle, contRef, ohlcv) {
        if (!list.length || !chart || !candle || !contRef || !ohlcv || !ohlcv.length) return;
        var layer = _measureSavedLayer(contRef);
        for (var i = 0; i < list.length; i++) {
            var m = list[i], els = _measureEls(m);
            // A chart rebuild wipes the container: put the elements back
            for (var j = 0; j < els.length; j++) if (els[j].parentNode !== layer) layer.appendChild(els[j]);
            var res = _computeMeasureResult(ohlcv, m.startTime, m.startPrice, m.endTime, m.endPrice);
            m.box = _measureRenderItem(chart, candle, contRef, m, res);
        }
    }

    // Topmost committed measurement under a click: on either line, on the dashed arrow, or on a label
    function _measureHitTest(list, contRef, clientX, clientY) {
        var cr = contRef.getBoundingClientRect();
        var x = clientX - cr.left, y = clientY - cr.top, tol = 5;
        for (var i = list.length - 1; i >= 0; i--) {
            var m = list[i], b = m.box;
            if (b) {
                var inX = x >= b.left - tol && x <= b.right + tol;
                if (inX && (Math.abs(y - b.top) <= tol || Math.abs(y - b.bottom) <= tol)) return m;
                if (Math.abs(x - b.midX) <= tol && y >= b.top - tol && y <= b.bottom + tol) return m;
            }
            var lbls = [m.tTop, m.tBot];
            for (var j = 0; j < lbls.length; j++) {
                var r = lbls[j].getBoundingClientRect();
                if (r.width && clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom) return m;
            }
        }
        return null;
    }

    function _measureSelect(list, item) {
        for (var i = 0; i < list.length; i++) {
            list[i].sel = (list[i] === item);
            _measureStyleSel(list[i]);
        }
    }

    function _measureRemoveEls(m) {
        var els = _measureEls(m);
        for (var j = 0; j < els.length; j++) if (els[j].parentNode) els[j].parentNode.removeChild(els[j]);
    }

    function _measureDeleteSelected(list) {
        for (var i = 0; i < list.length; i++) {
            if (list[i].sel) {
                var gone = list.splice(i, 1)[0];
                _measureRemoveEls(gone);
                if (gone.sym) _cdDropMs(gone.sym, gone);   // explicit delete only: also forget the saved copy
                return true;
            }
        }
        return false;
    }

    function _measureClearAll(list) {
        while (list.length) _measureRemoveEls(list.pop());
    }

    // Average bar spacing in seconds — used to project times outside the loaded data.
    function _measureStepSec(ohlcv) {
        var n = ohlcv.length;
        return n > 1 ? (ohlcv[n - 1].time - ohlcv[0].time) / (n - 1) : 86400;
    }
    // Bar index for a time, extended past both ends of the data: negative before the first
    // bar, >= length after the last. (_barIdxByTime returns -1 past the last bar.)
    function _measureIdxByTime(ohlcv, time) {
        var n = ohlcv.length;
        if (time > ohlcv[n - 1].time) return n - 1 + Math.round((time - ohlcv[n - 1].time) / _measureStepSec(ohlcv));
        if (time < ohlcv[0].time)     return Math.round((time - ohlcv[0].time) / _measureStepSec(ohlcv));
        return _barIdxByTime(ohlcv, time);
    }
    // "63 days" / "1 week" / "5 months" -- one wording for the live preview and the committed label.
    // The count includes BOTH end candles: res.barCount is the number of gaps between the two candles, so the
    // candles covered are barCount + 1 (9 candles side by side read "9 days"). A measurement inside a single
    // candle reads "1 day".
    function _measureLenLabel(res) {
        var n = res.barCount + 1;
        return n + ' ' + res.unit + (n === 1 ? '' : 's');
    }
    function _computeMeasureResult(ohlcv, startTime, startPrice, endTime, endPrice) {
        // bar count = |endIdx - startIdx| (TV-style: intervals between bars)
        var si = _measureIdxByTime(ohlcv, startTime);
        var ei = _measureIdxByTime(ohlcv, endTime);
        var barCount = Math.abs(ei - si);
        // The length is the candle count (barCount gaps + 1, see _measureLenLabel), shown in the chart's own unit: days on the daily chart (every bar
        // is a trading day -- no bars exist on weekends/holidays), weeks on weekly, months on monthly. The unit is
        // detected from the average bar spacing (~1.4 days daily, ~7 weekly, ~30 monthly), which keeps this
        // function independent of which chart calls it.
        var stepDays = _measureStepSec(ohlcv) / 86400;
        var unit = stepDays < 3 ? 'day' : (stepDays < 15 ? 'week' : 'month');
        var priceDelta = endPrice - startPrice;
        var pctDelta   = (priceDelta / Math.abs(startPrice)) * 100;
        return { startTime: startTime, startPrice: startPrice,
                 endTime: endTime, endPrice: endPrice,
                 startBarIdx: si, endBarIdx: ei,
                 barCount: barCount, unit: unit,
                 priceDelta: priceDelta, pctDelta: pctDelta };
    }

    function _measureGetTimeAtX(chart, ohlcv, lx) {
        var t = chart.timeScale().coordinateToTime(lx);
        if (t != null) return t;
        // Outside the loaded data. coordinateToTime returns null on BOTH sides (left of the first
        // bar as well as right of the last), so project from the logical index instead.
        var li = chart.timeScale().coordinateToLogical(lx);
        if (li == null) return null;
        var n = ohlcv.length, step = _measureStepSec(ohlcv);
        return li > n - 1 ? ohlcv[n - 1].time + (li - (n - 1)) * step
                          : ohlcv[0].time + li * step;
    }

    // ── Snapping + adjusting placed measurements ─────────────────────────────
    // Index of the candle a time belongs to: an exact time match, or the nearest candle within half a bar.
    // -1 when the time is outside the loaded candles (e.g. a future anchor), so nothing snaps there.
    function _measureBarAt(ohlcv, time) {
        var n = ohlcv ? ohlcv.length : 0;
        if (!n || time == null) return -1;
        var lo = 0, hi = n - 1;
        while (lo < hi) { var mid = (lo + hi) >> 1; if (ohlcv[mid].time < time) lo = mid + 1; else hi = mid; }
        var best = lo;
        if (lo > 0 && Math.abs(ohlcv[lo - 1].time - time) < Math.abs(ohlcv[lo].time - time)) best = lo - 1;
        return Math.abs(ohlcv[best].time - time) <= _measureStepSec(ohlcv) / 2 ? best : -1;
    }
    // Wick-tip snap. Looks at the high and the low of every candle within _MEASURE_SNAP_REACH candles of the cursor and
    // returns the ONE tip that is closest to the cursor ON SCREEN (pixel distance): { time, price, x, y }, or null when
    // the cursor is outside the loaded candles. The candle is part of the result, so a click just below a wick
    // can land on the neighbouring candle's tip if that tip is the one you are pointing at. (The old rule took the
    // candle under the cursor by x alone and then high-or-low by price, which missed the lowest wick of a base
    // whenever the cursor sat nearer the next candle.) Used only when an anchor is PLACED (click) or an adjustment is
    // RELEASED -- the live preview and the drag itself stay free. lx / ly are chart-container pixel coordinates.
    var _MEASURE_SNAP_REACH = 3;
    function _measureSnapTip(chart, candle, ohlcv, lx, ly) {
        if (!chart || !candle || !ohlcv || !ohlcv.length || lx == null || ly == null) return null;
        var c = _measureBarAt(ohlcv, _measureGetTimeAtX(chart, ohlcv, lx));   // candle under the cursor, -1 outside the data
        if (c < 0) return null;
        var ts = chart.timeScale(), best = null, bestD = Infinity;
        var from = Math.max(0, c - _MEASURE_SNAP_REACH), to = Math.min(ohlcv.length - 1, c + _MEASURE_SNAP_REACH);
        for (var i = from; i <= to; i++) {
            var b = ohlcv[i], x = ts.timeToCoordinate(b.time);
            if (x == null) continue;
            for (var k = 0; k < 2; k++) {
                var p = k ? b.low : b.high;
                if (typeof p !== 'number') continue;
                var y = candle.priceToCoordinate(p);
                if (y == null) continue;
                var d = Math.hypot(x - lx, y - ly);
                if (d < bestD) { bestD = d; best = { time: b.time, price: p, x: x, y: y }; }
            }
        }
        return best;
    }
    // Snap-target dot: a small ring on the wick tip a click would snap to. Shown while the measure tool is armed
    // (tool on, Shift held, or between the two clicks) and while dragging an end handle. Purely a pointer: it never
    // moves a measurement. One per chart container, kept in the same SVG layer as the placed measurements.
    function _measureSnapDot(contRef, create) {
        var d = contRef._msSnapDot;
        if (d && d.isConnected) return d;
        contRef._msSnapDot = null;
        if (!create) return null;
        d = _measureMakeEl(_measureSavedLayer(contRef), 'circle',
            { 'class': 'measure-snap-dot', 'r': '5', 'fill': 'none', 'stroke': 'var(--text-primary)', 'stroke-width': '2' });
        d.style.display = 'none';
        contRef._msSnapDot = d;
        if (!contRef._msSnapLeave) {
            contRef._msSnapLeave = true;
            contRef.addEventListener('mouseleave', function() { _measureHideSnapDot(contRef); });
        }
        return d;
    }
    function _measureShowSnapDot(contRef, tip) {
        if (!tip) { _measureHideSnapDot(contRef); return; }
        var d = _measureSnapDot(contRef, true);
        d.setAttribute('cx', tip.x); d.setAttribute('cy', tip.y);
        d.style.display = '';
    }
    function _measureHideSnapDot(contRef) {
        var d = contRef && _measureSnapDot(contRef, false);
        if (d) d.style.display = 'none';
    }
    // Hover hook (called from _onTrendMouseMoveCore): show / hide the dot for the current pointer position.
    function _measureUpdateSnapDot(evt, cfg) {
        var contRef = cfg.contRef;
        if (!contRef || !cfg.chart || !cfg.candle || !cfg.ohlcv || !cfg.ohlcv.length) return;
        var armed = !!(evt.shiftKey || cfg.getMeasureMode() || (cfg.getMeasurePhase && cfg.getMeasurePhase() === 1));
        if (!armed) { _measureHideSnapDot(contRef); return; }
        var r = contRef.getBoundingClientRect();
        var lx = evt.clientX - r.left, ly = evt.clientY - r.top;
        if (lx < 0 || ly < 0 || lx > r.width || ly > r.height) { _measureHideSnapDot(contRef); return; }
        // Over an end handle a click grabs the handle instead of placing an anchor, so no target is shown there
        if (cfg.measureList && cfg.measureList.length && _measureHandleHit(cfg.measureList, contRef, evt.clientX, evt.clientY)) {
            _measureHideSnapDot(contRef); return;
        }
        _measureShowSnapDot(contRef, _measureSnapTip(cfg.chart, cfg.candle, cfg.ohlcv, lx, ly));
    }
    // Inverse of _measureIdxByTime: the time of bar index idx, projected past either end of the data.
    function _measureTimeByIdx(ohlcv, idx) {
        var n = ohlcv.length;
        if (idx >= 0 && idx < n) return ohlcv[idx].time;
        var step = _measureStepSec(ohlcv);
        return idx >= n ? ohlcv[n - 1].time + (idx - (n - 1)) * step : ohlcv[0].time + idx * step;
    }
    // End handle under the cursor on a SELECTED measurement: { m, which: 'start' | 'end' } or null.
    function _measureHandleHit(list, contRef, clientX, clientY) {
        var cr = contRef.getBoundingClientRect();
        var x = clientX - cr.left, y = clientY - cr.top, HIT = 10;
        for (var i = list.length - 1; i >= 0; i--) {
            var m = list[i], b = m.box;
            if (!m.sel || !b) continue;
            if (Math.hypot(x - b.x2, y - b.y2) <= HIT) return { m: m, which: 'end' };
            if (Math.hypot(x - b.x1, y - b.y1) <= HIT) return { m: m, which: 'start' };
        }
        return null;
    }
    // Drag a placed measurement. mode: 'start' / 'end' = move that one anchor (it snaps to the closest wick tip
    // when you let go), 'move' = shift the whole measurement (same size, never snapped, so its measured
    // size stays exact). Nothing changes until the mouse has travelled a few pixels, so a plain click just selects.
    // The listeners belong to the drag, so no per-chart wiring is needed. On release the saved copy is replaced.
    function _measureBeginDrag(cfg, m, mode, evt) {
        var chart = cfg.chart, candle = cfg.candle, contRef = cfg.contRef, ohlcv = cfg.ohlcv;
        if (!chart || !candle || !contRef || !ohlcv || !ohlcv.length || !m.box) return;
        var r0    = contRef.getBoundingClientRect();
        var downX = evt.clientX, downY = evt.clientY;
        var orig  = { startTime: m.startTime, startPrice: m.startPrice, endTime: m.endTime, endPrice: m.endPrice };
        var downP = candle.coordinateToPrice(downY - r0.top);
        var downL = chart.timeScale().coordinateToLogical(downX - r0.left);
        // Keep the anchor where it was grabbed (the handle is ~10px wide) so it doesn't jump to the cursor
        var offX = 0, offY = 0;
        if (mode === 'start') { offX = m.box.x1 - (downX - r0.left); offY = m.box.y1 - (downY - r0.top); }
        if (mode === 'end')   { offX = m.box.x2 - (downX - r0.left); offY = m.box.y2 - (downY - r0.top); }
        var moved = false, rafId = null, lastX = downX, lastY = downY, anchorX = null, anchorY = null;

        function redraw() {
            var res = _computeMeasureResult(ohlcv, m.startTime, m.startPrice, m.endTime, m.endPrice);
            m.box = _measureRenderItem(chart, candle, contRef, m, res);
        }
        function apply() {
            if (!m.topLine.isConnected) return;          // the chart was rebuilt under the drag
            var r  = contRef.getBoundingClientRect();
            var lx = lastX - r.left, ly = lastY - r.top;
            if (mode === 'move') {
                var P = candle.coordinateToPrice(ly), L = chart.timeScale().coordinateToLogical(lx);
                if (P == null || L == null || downP == null || downL == null) return;
                var dBars = Math.round(L - downL), dP = P - downP;
                m.startTime  = _measureTimeByIdx(ohlcv, _measureIdxByTime(ohlcv, orig.startTime) + dBars);
                m.endTime    = _measureTimeByIdx(ohlcv, _measureIdxByTime(ohlcv, orig.endTime)   + dBars);
                m.startPrice = orig.startPrice + dP;
                m.endPrice   = orig.endPrice   + dP;
            } else {
                anchorX = lx + offX; anchorY = ly + offY;      // where the handle is on screen: this is what snaps on release
                var aP = candle.coordinateToPrice(anchorY), aT = _measureGetTimeAtX(chart, ohlcv, anchorX);
                if (aP == null || aT == null) return;
                if (mode === 'start') { m.startTime = aT; m.startPrice = aP; }
                else                  { m.endTime   = aT; m.endPrice   = aP; }
                _measureShowSnapDot(contRef, _measureSnapTip(chart, candle, ohlcv, anchorX, anchorY));   // where it will land on release
            }
            redraw();
        }
        function onMove(e) {
            lastX = e.clientX; lastY = e.clientY;
            if (!moved) {
                if (Math.hypot(lastX - downX, lastY - downY) < 3) return;
                moved = true;
                contRef.style.cursor = 'grabbing';
            }
            if (rafId) return;
            rafId = requestAnimationFrame(function() { rafId = null; apply(); });
        }
        function onUp(e) {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup',   onUp);
            if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
            contRef.style.cursor = '';
            _measureHideSnapDot(contRef);
            if (!moved || !m.topLine.isConnected) return;   // plain click, or the measurement is gone: nothing to change
            lastX = e.clientX; lastY = e.clientY;
            apply();                                        // final position (this also re-shows the snap dot)
            _measureHideSnapDot(contRef);
            if (mode !== 'move' && anchorX != null) {            // closest wick tip to the handle: both its candle and its price
                var tip = _measureSnapTip(chart, candle, ohlcv, anchorX, anchorY);
                if (tip && mode === 'start') { m.startTime = tip.time; m.startPrice = tip.price; }
                if (tip && mode === 'end')   { m.endTime   = tip.time; m.endPrice   = tip.price; }
            }
            if (m.startTime === m.endTime && m.startPrice === m.endPrice) {   // both ends on one point: a zero-size measurement, keep the original
                m.startTime = orig.startTime; m.startPrice = orig.startPrice;
                m.endTime   = orig.endTime;   m.endPrice   = orig.endPrice;
            }
            redraw();
            var changed = m.startTime !== orig.startTime || m.startPrice !== orig.startPrice ||
                          m.endTime   !== orig.endTime   || m.endPrice   !== orig.endPrice;
            if (changed && m.sym) { _cdDropMs(m.sym, orig); _cdAddMs(m.sym, m); }   // replace the saved copy
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup',   onUp);
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── Shared cell chart renderer ────────────────────────────────────────────
    function renderLwMcCellChart(container, ohlcv) {
        container.innerHTML = '';
        if (!window.LightweightCharts || !ohlcv || !ohlcv.length) {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:11px;">No data</div>';
            return null;
        }
        var chart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted') },
            grid:    { vertLines: { color: themeColor('mc-cell-grid') }, horzLines: { color: themeColor('mc-cell-grid') } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
            rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted'), scaleMargins: { top: 0.06, bottom: 0.22 } },
            timeScale: { borderColor: themeColor('bg-surface'), timeVisible: false, rightOffset: 1 },
            handleScroll: false, handleScale: false,
        });
        var candle = chart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'), borderVisible: false,
            wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
            priceLineVisible: false, lastValueVisible: true,
        });
        candle.setData(ohlcv);

        // Active MAs — mirrors fullscreen MA toggle state (EMA8 + SMA150 excluded: too noisy in multichart)
        Object.keys(_mcFsActiveMas).forEach(function(key) {
            if (!_mcFsActiveMas[key]) return;
            if (key === 'EMA8' || key === 'SMA150') return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = chart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
            s.setData(_calcMA(ohlcv, key));
        });

        // Volume
        var vol = chart.addSeries(LightweightCharts.HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
        chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
        vol.setData(ohlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
        }));

        // Volume 50-SMA — same scale as volume bars
        (function() {
            var period = 50, volSmaData = [];
            for (var i = period - 1; i < ohlcv.length; i++) {
                var sum = 0;
                for (var j = i - (period - 1); j <= i; j++) sum += (ohlcv[j].volume || 0);
                volSmaData.push({ time: ohlcv[i].time, value: sum / period });
            }
            var volMa = chart.addSeries(LightweightCharts.LineSeries, {
                color: '#1848cc', lineWidth: 1, priceScaleId: 'vol',
                priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
            });
            volMa.setData(volSmaData);
        })();

        var n = ohlcv.length;
        chart.timeScale().setVisibleLogicalRange({ from: n - 65, to: n });

        // OHLC legend
        var leg = document.createElement('div');
        leg.style.cssText = 'position:absolute;top:0;left:6px;z-index:10;font-size:10px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-muted-2);pointer-events:none;line-height:1.5;background:var(--bg-page-alpha-2);padding:2px 5px;border-radius:3px;';
        container.style.position = 'relative';
        container.appendChild(leg);
        function fp(v) { return v != null ? v.toFixed(2) : '—'; }
        function fv(v) { return v==null?'—':v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':v.toFixed(0); }
        chart.subscribeCrosshairMove(function(p) {
            if (!p.time || !p.seriesData || !p.seriesData.size) { leg.innerHTML = ''; return; }
            var d = p.seriesData.get(candle); if (!d) { leg.innerHTML = ''; return; }
            var cl = d.close >= d.open ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
            var vd = p.seriesData.get(vol);
            leg.innerHTML = '<span style="color:var(--text-muted)">O</span><span style="color:'+cl+'">'+fp(d.open)+'</span> <span style="color:var(--text-muted)">H</span><span style="color:'+cl+'">'+fp(d.high)+'</span> <span style="color:var(--text-muted)">L</span><span style="color:'+cl+'">'+fp(d.low)+'</span> <span style="color:var(--text-muted)">C</span><span style="color:'+cl+'">'+fp(d.close)+'</span>'+(vd?'  <span style="color:var(--border-muted)">V</span><span style="color:var(--text-muted)">'+fv(vd.value)+'</span>':'');
        });
        return { chart: chart, candle: candle, vol: vol, ohlcv: ohlcv };
    }

    // ── Push live intraday price into a rendered multichart cell ──────────────
    function _updateMcLiveCandle(ticker, price, dayHigh, dayLow, widgetsObj) {
        var inst = widgetsObj && widgetsObj[ticker];
        if (!inst || !inst.candle || !inst.ohlcv || !inst.ohlcv.length) return;
        // Weekly/Monthly tiles: fold into the current period's bar instead of
        // appending a one-day candle to a W/M series.
        if (inst.tf && inst.tf !== 'D') {
            var _wm = _mcApplyLiveWM(inst.ohlcv, inst.tf, price, dayHigh, dayLow, true);
            if (!_wm) return;
            try { inst.candle.update(_wm); } catch(e) {}
            if (inst.vol) {
                try { inst.vol.update({ time: _wm.time, value: _wm.volume, color: price >= _wm.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') }); } catch(e) {}
            }
            return;
        }
        var d = new Date();
        // Use noon UTC (midnight UTC + 43200s) so todayTs matches the noon-UTC stamps
        // written by fetchMcOhlcv and stays within the correct calendar day for UTC-N zones.
        var todayTs = Math.floor(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000) + 43200;
        var last = inst.ohlcv[inst.ohlcv.length - 1];
        var lastDayTs = Math.floor(last.time / 86400) * 86400 + 43200;
        var open, high, low, volume;
        if (lastDayTs === todayTs) {
            open   = last.open;
            high   = dayHigh != null ? Math.max(last.high, dayHigh, price) : Math.max(last.high, price);
            low    = dayLow  != null ? Math.min(last.low,  dayLow,  price) : Math.min(last.low,  price);
            volume = last.volume;
            last.high  = high;
            last.low   = low;
            last.close = price;
        } else {
            // Only create a new bar during market hours — prevents a phantom candle
            // appearing after close.
            if (!wlIsMarketOpen()) return;
            open = high = low = price;
            volume = 0;
            inst.ohlcv.push({ time: todayTs, open: open, high: high, low: low, close: price, volume: volume });
        }
        try { inst.candle.update({ time: todayTs, open: open, high: high, low: low, close: price, volume: volume }); } catch(e) {}
        if (inst.vol) {
            try { inst.vol.update({ time: todayTs, value: volume, color: price >= open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') }); } catch(e) {}
        }
    }

    function _destroyMcWidgets(widgets) {
        Object.keys(widgets).forEach(function(sym) {
            var inst = widgets[sym];
            if (inst && inst.chart) { try { inst.chart.remove(); } catch(e) {} }
        });
    }

    // ── Shared multichart grid builder ────────────────────────────────────────
    function _buildLwMcGrid(grid, tickers, tf, cols, widgetsObj, contextKey) {
        _destroyMcWidgets(widgetsObj);
        Object.keys(widgetsObj).forEach(function(k) { delete widgetsObj[k]; });
        grid.innerHTML = '';
        grid.style.gridTemplateColumns = 'repeat(' + cols + ', 1fr)';

        _mcRenderTokens[contextKey]++;
        var token = _mcRenderTokens[contextKey];

        tickers.forEach(function(sym) {
            var cell = document.createElement('div');
            cell.className = 'mc-cell';
            cell.setAttribute('data-sym', sym);

            var flag = document.createElement('button');
            flag.className = 'mc-cell-flag' + (wlIsFlagged(sym) ? ' flagged' : '');
            flag.textContent = '⚑'; flag.title = 'Add to Flagged';
            flag.addEventListener('click', function(e) { e.stopPropagation(); wlFlagTicker(sym, flag); });

            var hdr = buildMcCellHeader(sym, flag);

            var hint = document.createElement('div');
            hint.className = 'mc-cell-hint'; hint.textContent = 'click to expand';

            var chartDiv = document.createElement('div');
            chartDiv.style.cssText = 'width:100%;flex:1;min-height:0;position:relative;overflow:hidden;';
            chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:11px;">Loading…</div>';

            var overlay = document.createElement('div');
            overlay.className = 'mc-cell-overlay';
            overlay.addEventListener('click', function(e) {
                e.stopPropagation();
                if (failed) { attemptLoad(); return; }
                _mcFsTf = tf;
                openChartModal(sym);
            });
            overlay.addEventListener('contextmenu', function(e) {
                e.preventDefault(); e.stopPropagation();
                var fakeBtn = {
                    getAttribute: function(a) { return a === 'data-ticker' ? sym : null; },
                    getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                    _wlNoSwitch: true
                };
                wlOpenPicker(fakeBtn, e, false);
            });

            cell.appendChild(hdr); cell.appendChild(hint); cell.appendChild(overlay); cell.appendChild(chartDiv);
            grid.appendChild(cell);

            var rendered = false;
            var failed = false;

            // Background retry schedule, tried only after the queue's own quick
            // 1s/2s/4s retries (in doFetch) are already exhausted. Spread further
            // apart and outside the concurrency queue so a slow-to-recover ticker
            // never occupies one of the MC_FETCH_LIMIT slots while waiting.
            var MC_BG_RETRY_DELAYS = [10000, 20000, 40000, 60000]; // ~2min total

            function attemptLoad(bgAttempt) {
                bgAttempt = bgAttempt || 0;
                // Grid tiles are hidden behind the fullscreen overlay when it's
                // open, and every fetch shares one global pacer/429-cooldown
                // (yahoo-proxy-pace.js) — so a background retry here can eat the
                // budget a deliberate single-chart open needs. Stand down and
                // recheck shortly instead of spending this retry attempt while it's up.
                if (_mcFsIsOpen()) {
                    setTimeout(function() {
                        if (_mcRenderTokens[contextKey] !== token) return;
                        if (!document.body.contains(chartDiv) || chartDiv.offsetParent === null) return;
                        attemptLoad(bgAttempt);
                    }, 1000);
                    return;
                }
                failed = false;
                chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:11px;">Loading…</div>';
                fetchMcOhlcv(sym, tf, true).then(function(ohlcv) {
                    if (_mcRenderTokens[contextKey] !== token) return;
                    if (ohlcv === null) {
                        if (bgAttempt < MC_BG_RETRY_DELAYS.length) {
                            // Quietly try again later — covers the common case
                            // (transient rate limit) with no click required.
                            setTimeout(function() {
                                if (_mcRenderTokens[contextKey] !== token) return; // grid moved on, abandon
                                // Tab switches in this app hide the grid via display:none
                                // rather than removing it, so the render token alone won't
                                // catch "you navigated away." offsetParent is null whenever
                                // an element (or an ancestor) is display:none, so this is
                                // what actually stops retries once the tile's tab isn't visible.
                                if (!document.body.contains(chartDiv) || chartDiv.offsetParent === null) return;
                                attemptLoad(bgAttempt + 1);
                            }, MC_BG_RETRY_DELAYS[bgAttempt]);
                            return;
                        }
                        // ~2 minutes of retries exhausted — genuinely persistent
                        // failure. Manual retry is now the fallback, not the default.
                        failed = true;
                        chartDiv.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:4px;color:var(--text-muted-2);font-size:11px;"><span>Failed to load</span><span style="text-decoration:underline;">Click to retry</span></div>';
                        return;
                    }
                    try {
                        var inst = renderLwMcCellChart(chartDiv, ohlcv);
                        rendered = true;
                        if (inst) { inst.tf = tf; widgetsObj[sym] = inst; }
                    } catch(e) {
                        chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:11px;">Error</div>';
                    }
                }).catch(function() {
                    chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:11px;">Error</div>';
                });
            }

            var obs = new IntersectionObserver(function(entries) {
                if (!entries[0].isIntersecting || rendered) return;
                obs.disconnect();
                attemptLoad();
            // rootMargin gives a ~600px lookahead below the viewport so a
            // chart starts fetching just before it's scrolled into view,
            // instead of only once it's actually visible.
            }, { threshold: 0.05, rootMargin: '600px 0px' });
            obs.observe(cell);
        });

        // Deliberately no pre-warm-everything loop here. A broad scan can
        // hand this function 100+ tickers; only ~12 are ever visible at once
        // (see mcCols), so eagerly fetching all of them up front was firing
        // dozens of concurrent 10-year-history requests for charts nobody
        // had scrolled to yet — the direct cause of the 429 flood. The
        // IntersectionObserver above (with its lookahead margin) is the only
        // thing that should be triggering fetches now.
    }

    // ── Fullscreen LW chart ────────────────────────────────────────────────────
    // Crosshair date label — "Wed 22 Apr '26" (3-letter weekday, 3-letter month,
    // apostrophe + 2-digit year). Bars are stored at UTC noon (see _noonTs above)
    // specifically so reading UTC date parts here always lands on the right
    // calendar day regardless of the viewer's local timezone.
    function _mcLwCrosshairDateFmt(time) {
        var d      = new Date(time * 1000);
        var days   = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
        var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return days[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' + months[d.getUTCMonth()] + ' \'' + String(d.getUTCFullYear()).slice(-2);
    }

    function _addFsVwap(anchorIdx) {
        if (!_mcFsChart || !_mcFsOhlcv.length) return;
        var color = _AVWAP_COLOR;
        var data  = _calcAVWAP(_mcFsOhlcv, anchorIdx);
        if (!data.length) return;
        var s = _mcFsChart.addSeries(LightweightCharts.LineSeries, { color: color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
        s.setData(data);
        var dataMap = new Map(data.map(function(d) { return [d.time, d.value]; }));
        _mcFsVwapSeries.push({ series: s, anchor: anchorIdx, color: color, dataMap: dataMap });
    }

    // Future bar slots. Daily-like series (bars at most ~5 days apart) advance by TRADING days: weekends are skipped,
    // exactly as the chart spaces real bars and exactly as the alert engine counts slots (alerts.js _alSlotOfDay). The
    // old `last.time + n * (last.time - prev.time)` made one "bar" three days long whenever the last two bars straddled
    // a weekend, which silently shifted every future-anchored line (and the alert evaluated on it).
    // Weekly / monthly series keep calendar steps (a week is still 5 weekdays per bar).
    function _mcIsDailyLike(ohlcv) {
        return ohlcv.length < 2 || (ohlcv[ohlcv.length - 1].time - ohlcv[ohlcv.length - 2].time) <= 5.5 * 86400;
    }
    // Timestamp of the slot `n` bars past the last bar.
    function _mcFutureTime(ohlcv, n) {
        var last = ohlcv[ohlcv.length - 1];
        var prev = ohlcv[ohlcv.length - 2] || last;
        if (!_mcIsDailyLike(ohlcv)) return last.time + n * (last.time - prev.time);
        var day = Math.floor(last.time / 86400);
        for (var k = 0; k < n; ) { day++; var w = (day + 4) % 7; if (w !== 0 && w !== 6) k++; }
        return day * 86400 + (last.time - Math.floor(last.time / 86400) * 86400);   // keep the bars' own time-of-day (noon UTC)
    }

    // Converts a time to a pixel X coordinate. Future timestamps have no entry in LWC's internal time scale, so they
    // are placed by counting slots past the last bar (trading days for daily-like series).
    function _mcFsTimeToX(chart, ohlcv, time) {
        var x = chart.timeScale().timeToCoordinate(time);
        if (x !== null) return x;
        if (ohlcv.length < 2) return null;
        var last  = ohlcv[ohlcv.length - 1];
        var prev  = ohlcv[ohlcv.length - 2];
        var lastX = chart.timeScale().timeToCoordinate(last.time);
        var prevX = chart.timeScale().timeToCoordinate(prev.time);
        if (lastX == null || prevX == null) return null;
        if (_mcIsDailyLike(ohlcv)) {
            var d0 = Math.floor(last.time / 86400), d1 = Math.floor(time / 86400), slots = 0;
            for (var d = d0 + 1; d <= d1; d++) { var w = (d + 4) % 7; if (w !== 0 && w !== 6) slots++; }
            var wEnd = (d1 + 4) % 7;
            if (d1 > d0 && (wEnd === 0 || wEnd === 6)) slots += 1;     // a weekend date sits on the next Monday's slot
            return lastX + slots * (lastX - prevX);
        }
        var pxPerSec = (lastX - prevX) / (last.time - prev.time);
        return lastX + pxPerSec * (time - last.time);
    }

    // Draw a trendline through two bar-index/price anchors using a v5 canvas
    // primitive.  The trendline object stores leftP/rightP for hit-testing and
    // a `selected` flag that the renderer reads to draw the highlight state.
    // Builds the canvas-primitive trendline object and attaches it to the
    // candle series. Shared by fullscreen/watchlist/alerts (alerts.js calls
    // this cross-file, same pattern as the other shared cores above).
    // opts.extend: also draw a dashed continuation of the line to the right edge. Used for alert-backed lines:
    // the alert is evaluated on that continuation at today's bar, so it is the level actually being monitored
    // (the old finite segment hid it, so an alert could fire on a line you could not see).
    // Returns the drawing object (callers may ignore it).
    function _addTrendlineCore(p1, p2, chart, candle, ohlcv, trendlines, opts) {
        if (!chart || !candle || !ohlcv.length) return null;
        var refChart  = chart;
        var refSeries = candle;

        // Normalise so leftP is always the earlier anchor (by time, supports future timestamps)
        var leftP  = p1.time <= p2.time ? p1 : p2;
        var rightP = p1.time <= p2.time ? p2 : p1;

        // Create the object first so the primitive closes over it
        var tlObj = { p1: p1, p2: p2, leftP: leftP, rightP: rightP, selected: false, requestUpdate: null,
                      extend: !!(opts && opts.extend),
                      // Dotted is a drawing style only: the line is the same object with the same anchors, hit-test,
                      // selection, drag and alert behaviour as a solid one — just stroked with round dots.
                      dotted: !!(opts && opts.dotted),
                      // The line's identity as the alert store knows it. Dragging an anchor moves the matching
                      // alert from these points to the new ones (see _onTrendAnchorDragEndCore).
                      alertRef: { l: { time: leftP.time, price: leftP.price }, r: { time: rightP.time, price: rightP.price } } };

        var primitive = {
            attached: function(param) {
                tlObj.requestUpdate = function() { try { param.requestUpdate(); } catch(e) {} };
                param.requestUpdate();
            },
            paneViews: function() {
                return [{
                    renderer: function() {
                        return {
                            draw: function(target) {
                                if (tlObj.dragging) return; // SVG owns the preview during drag
                                // Use helper that extrapolates for future timestamps
                                var x1 = _mcFsTimeToX(refChart, ohlcv, tlObj.leftP.time);
                                var x2 = _mcFsTimeToX(refChart, ohlcv, tlObj.rightP.time);
                                var y1 = refSeries.priceToCoordinate(tlObj.leftP.price);
                                var y2 = refSeries.priceToCoordinate(tlObj.rightP.price);
                                if (x1 == null || x2 == null || y1 == null || y2 == null) return;
                                target.useBitmapCoordinateSpace(function(scope) {
                                    var ctx = scope.context;
                                    var rx  = scope.horizontalPixelRatio;
                                    var ry  = scope.verticalPixelRatio;
                                    var bx1 = x1 * rx, by1 = y1 * ry;
                                    var bx2 = x2 * rx, by2 = y2 * ry;
                                    ctx.save();
                                    // Main line — colour unchanged whether selected or not
                                    ctx.beginPath();
                                    ctx.moveTo(bx1, by1);
                                    ctx.lineTo(bx2, by2);
                                    ctx.strokeStyle = _TRENDLINE_COLOR();
                                    ctx.lineWidth   = 1.5 * rx;
                                    if (tlObj.dotted) { ctx.lineWidth = 2.5 * rx; ctx.lineCap = 'round'; ctx.setLineDash([0.1 * rx, 5.9 * rx]); } // dotted: 2.5px round dots, 6px apart
                                    ctx.stroke();
                                    if (tlObj.dotted) { ctx.setLineDash([]); ctx.lineCap = 'butt'; ctx.lineWidth = 1.5 * rx; } // don't leak the dots into the continuation / anchor handles
                                    // Dashed continuation to the right edge (alert-backed lines only). Bars are evenly
                                    // spaced per trading day, so a straight pixel line here is the same line the alert
                                    // evaluates in trading-day slots.
                                    if (tlObj.extend && _TRENDLINE_SHOW_CONTINUATION) {
                                        try {
                                            var xr = scope.bitmapSize.width;
                                            if (bx2 !== bx1 && xr > bx2) {
                                                ctx.save();
                                                ctx.setLineDash([6 * rx, 5 * rx]);
                                                ctx.globalAlpha = 0.65;
                                                ctx.beginPath();
                                                ctx.moveTo(bx2, by2);
                                                ctx.lineTo(xr, by2 + (by2 - by1) / (bx2 - bx1) * (xr - bx2));
                                                ctx.strokeStyle = _TRENDLINE_COLOR();
                                                ctx.lineWidth   = 1.25 * rx;
                                                ctx.stroke();
                                                ctx.restore();
                                            }
                                        } catch (e) { /* the continuation must never break the main line */ }
                                    }
                                    // Anchor dots only when selected
                                    if (tlObj.selected) {
                                        [[bx1, by1], [bx2, by2]].forEach(function(pt) {
                                            ctx.beginPath();
                                            ctx.arc(pt[0], pt[1], 4.5 * rx, 0, Math.PI * 2);
                                            // Anchor handles deliberately use text-emphasis, NOT the (lighter) trendline colour:
                                            // they are what you grab to adjust a line, so they must stay high-contrast in light mode.
                                            ctx.fillStyle   = themeColor('text-emphasis');
                                            ctx.fill();
                                            ctx.strokeStyle = themeColor('bg-page');
                                            ctx.lineWidth   = 1.5 * rx;
                                            ctx.stroke();
                                        });
                                    }
                                    ctx.restore();
                                });
                            }
                        };
                    }
                }];
            }
        };

        tlObj.primitive = primitive;
        refSeries.attachPrimitive(primitive);
        trendlines.push(tlObj);
        return tlObj;
    }

    function _addFsTrendline(p1, p2, extend, dotted) {
        return _addTrendlineCore(p1, p2, _mcFsChart, _mcFsCandle, _mcFsOhlcv, _mcFsTrendlines, { extend: !!extend, dotted: !!dotted });
    }

    // Index of an alert's AVWAP anchor in a chart's own bar array. Daily: the bar on/after the anchor day.
    // Weekly/monthly: the bar whose PERIOD contains it, so it resolves whatever day Yahoo stamped that bar with
    // (the old strict-equality match silently failed whenever a stamp landed on a non-trading day).
    function _chartAnchorIdx(ohlcv, tf, anchorUnix) {
        if (!ohlcv || !ohlcv.length || anchorUnix == null) return -1;
        var i;
        if (tf === 'W' || tf === 'M') {
            var key = _mcPeriodKey(anchorUnix, tf);
            for (i = 0; i < ohlcv.length; i++) if (_mcPeriodKey(ohlcv[i].time, tf) === key) return i;
            return -1;
        }
        var day = Math.floor(anchorUnix / 86400);
        if (Math.floor(ohlcv[0].time / 86400) > day) return -1;
        for (i = 0; i < ohlcv.length; i++) if (Math.floor(ohlcv[i].time / 86400) >= day) return i;
        return -1;
    }

    // Deleting a drawn line used to leave its alert armed: the alert kept monitoring a line you had deleted, and the
    // line was redrawn from the alert store the next time the chart opened. These ask the alert engine to delete the
    // alert(s) behind the line as well (it confirms first, unless "don't ask again" was ticked), then run `remove`.
    // `done` also drops the line's SAVED copy (see "Saved chart drawings" below). It only runs once the delete goes
    // through, so cancelling the "Delete line and alert?" dialog leaves the saved line alone.
    function _deleteTrendlineWithAlerts(sym, tl, remove) {
        var done = function() { remove(); if (tl && tl.alertRef) _cdDropTl(sym, tl.alertRef); };
        if (!tl || !tl.alertRef || !window.alDeleteLineAlerts) { done(); return; }
        window.alDeleteLineAlerts(sym, 'trendline', tl.alertRef, done);
    }
    function _deleteVwapWithAlerts(sym, ohlcv, tf, entry, remove) {
        var done = function() { remove(); if (entry) _cdDropAv(sym, ohlcv, tf, entry.anchor); };
        if (!entry || !window.alDeleteLineAlerts) { done(); return; }
        window.alDeleteLineAlerts(sym, 'avwap', { ohlcv: ohlcv, tf: tf, idx: entry.anchor }, done);
    }

    // Draws the alert-backed trendlines and AVWAPs onto a chart so they're visible when reviewing it.
    // One drawing per distinct line (an "above" and a "below" alert on the same line share one), and AVWAP anchors
    // are resolved on THIS chart's timeframe. AVWAP alerts used to be restored on the alerts chart only.
    function _restoreAlertLines(sym, tf, ohlcv, addTrendline, addVwap) {
        if (window.alGetTrendlineAlerts) {
            var seenTl = {};
            window.alGetTrendlineAlerts(sym).forEach(function(a) {
                if (!a.p1 || !a.p2) return;
                var k = a.p1.unix + '@' + a.p1.price + '|' + a.p2.unix + '@' + a.p2.price;
                if (seenTl[k]) return;
                seenTl[k] = true;
                addTrendline(a.p1, a.p2, true);
            });
        }
        if (window.alGetAvwapAlerts) {
            var seenAv = {};
            window.alGetAvwapAlerts(sym).forEach(function(a) {
                var u = (a.anchorUnix != null) ? a.anchorUnix : (typeof a.anchorTime === 'number' ? a.anchorTime : null);
                var idx = _chartAnchorIdx(ohlcv, tf, u);
                if (idx < 0 || seenAv[idx]) return;
                seenAv[idx] = true;
                addVwap(idx);
            });
        }
    }

    // ── Saved chart drawings (fullscreen + watchlist charts) ───────────────────────────────────────────
    // Trendlines and AVWAPs drawn by hand used to vanish on a timeframe change, a symbol change or closing the chart;
    // only alert-backed lines came back (_restoreAlertLines). They are now kept per ticker and redrawn the next time
    // that ticker's chart opens, on any timeframe.
    //   store = { TICKER: { tl: [ { l: {time, price}, r: {time, price}, dotted } ], av: [ anchorUnixSeconds ],
    //                       ms: [ { a: {time, price}, b: {time, price} } ] } }      (ms = committed measurements, see below)
    //  - Trendlines are saved by their two anchors (absolute time + price), so they don't depend on the timeframe.
    //  - AVWAPs are saved by the anchor bar's TIMESTAMP, never its index: an index only means something on the
    //    timeframe it was placed on. Restore resolves the timestamp on the chart being opened (_chartAnchorIdx). An
    //    anchor that doesn't resolve there (older than the loaded history) is not drawn but is KEPT in the store.
    //  - Changes are applied one entry at a time (add / move / drop). The store is never rebuilt from what is on
    //    screen: a snapshot would silently delete every saved line the current timeframe can't show.
    //  - The alerts side-panel chart neither draws nor saves hand-drawn trendlines / AVWAPs; it still restores alert-backed
    //    lines only. It DOES save and restore measurements (_cdAddMs / _cdRestoreMs).
    // Persistence: KV (kvGet/kvSet, like alerts) plus an immediate localStorage mirror. The whole store is ONE key and
    // KV writes are coalesced into one trailing write, because the Worker's KV has a small daily write allowance. The
    // blob carries a timestamp and the NEWER of KV / localStorage wins at load, so a KV write that never landed (tab
    // closed, Worker unreachable, origin not allowed) can't roll back a newer local copy.
    var _CD_KEY = 'chart_drawings', _CD_LS_KEY = 'chart_drawings_ls';
    var _cdStore = null;                 // loaded store, or null until the first load finishes
    var _cdStamp = 0;                    // ms timestamp of the last change
    var _cdLoadP = null;
    var _cdPending = [];                 // changes made before the first load finished; replayed on top of it
    var _cdKvTimer = null, _cdKvDirty = false;

    function _cdParse(raw) {
        try {
            var o = (typeof raw === 'string') ? JSON.parse(raw) : raw;
            if (o && typeof o === 'object' && o.d && typeof o.d === 'object') return { t: +o.t || 0, d: o.d };
        } catch (e) {}
        return null;
    }
    function _cdBlob() { return JSON.stringify({ v: 1, t: _cdStamp, d: _cdStore }); }
    function _cdHasData(d) { return !!d && Object.keys(d).length > 0; }

    function _cdFlushKv() {
        if (_cdKvTimer) { clearTimeout(_cdKvTimer); _cdKvTimer = null; }
        if (!_cdKvDirty || !_cdStore) return;
        _cdKvDirty = false;
        try {
            if (typeof kvSet !== 'function') return;
            var p = kvSet(_CD_KEY, _cdBlob());
            if (p && typeof p.catch === 'function') p.catch(function() { _cdKvDirty = true; });
        } catch (e) { _cdKvDirty = true; }
    }
    function _cdCommit() {
        _cdStamp = Date.now();
        try { localStorage.setItem(_CD_LS_KEY, _cdBlob()); } catch (e) {}
        _cdKvDirty = true;
        if (!_cdKvTimer) _cdKvTimer = setTimeout(_cdFlushKv, 2000);
    }
    window.addEventListener('pagehide', _cdFlushKv);

    function _cdLoad() {
        if (_cdLoadP) return _cdLoadP;
        var local = null;
        try { local = _cdParse(localStorage.getItem(_CD_LS_KEY)); } catch (e) {}
        var kvP;
        try { kvP = (typeof kvGet === 'function') ? Promise.resolve(kvGet(_CD_KEY)) : Promise.resolve(null); }
        catch (e) { kvP = Promise.resolve(null); }
        // A KV read that never answers must not block drawing or saving: fall back to the local copy after 4s.
        var timeout = new Promise(function(res) { setTimeout(function() { res(null); }, 4000); });
        _cdLoadP = Promise.race([kvP.catch(function() { return null; }), timeout]).then(function(raw) {
            var remote = _cdParse(raw);
            var pick = (remote && (!local || remote.t >= local.t)) ? remote : local;
            _cdStore = pick ? pick.d : {};
            _cdStamp = pick ? pick.t : 0;
            if (_cdPending.length) {
                var q = _cdPending; _cdPending = [];
                var changed = false;
                q.forEach(function(fn) { try { changed = fn(_cdStore) || changed; } catch (e) {} });
                if (changed) _cdCommit();
            } else if (pick === local && local && _cdHasData(local.d) && (!remote || local.t > remote.t)) {
                _cdKvDirty = true;                       // local is newer than KV (an earlier KV write was lost): repair once
                if (!_cdKvTimer) _cdKvTimer = setTimeout(_cdFlushKv, 2000);
            } else if (pick === remote) {
                try { localStorage.setItem(_CD_LS_KEY, _cdBlob()); } catch (e) {}   // keep the local mirror fresh
            }
            return _cdStore;
        });
        return _cdLoadP;
    }
    // fn(store) applies one change and returns true if it changed anything.
    function _cdMutate(fn) {
        if (_cdStore) { var ch = false; try { ch = fn(_cdStore); } catch (e) {} if (ch) _cdCommit(); return; }
        _cdPending.push(fn);
        _cdLoad();
    }

    function _cdNum(v) { return (typeof v === 'number' && isFinite(v)) ? v : null; }
    function _cdAnch(p) { return (p && _cdNum(p.time) != null && _cdNum(p.price) != null) ? { time: p.time, price: p.price } : null; }
    function _cdSamePt(a, b) { return a.time === b.time && a.price === b.price; }
    // Same line whichever end is called "left" (a drag can carry an anchor past the other one).
    function _cdSameLine(a, b) {
        return (_cdSamePt(a.l, b.l) && _cdSamePt(a.r, b.r)) || (_cdSamePt(a.l, b.r) && _cdSamePt(a.r, b.l));
    }
    function _cdPrune(store, sym) {
        var e = store[sym];
        if (e && !(e.tl && e.tl.length) && !(e.av && e.av.length) && !(e.ms && e.ms.length)) delete store[sym];
    }
    function _cdEntry(store, sym) {
        var e = store[sym] || (store[sym] = {});
        if (!e.tl) e.tl = [];
        if (!e.av) e.av = [];
        if (!e.ms) e.ms = [];
        return e;
    }

    // A hand-drawn trendline was committed.
    function _cdAddTl(sym, tl) {
        var a = tl && _cdAnch(tl.leftP), b = tl && _cdAnch(tl.rightP);
        if (!sym || !a || !b) return;
        var line = a.time <= b.time ? { l: a, r: b } : { l: b, r: a };
        line.dotted = !!tl.dotted;
        _cdMutate(function(store) {
            var e = _cdEntry(store, sym);
            if (e.tl.some(function(x) { return _cdSameLine(x, line); })) { _cdPrune(store, sym); return false; }
            e.tl.push(line);
            return true;
        });
    }
    // A drawn trendline was deleted. `ref` = its alertRef ({l, r} anchors as last committed).
    function _cdDropTl(sym, ref) {
        if (!sym || !ref || !ref.l || !ref.r) return;
        _cdMutate(function(store) {
            var e = store[sym];
            if (!e || !e.tl) return false;
            var n = e.tl.length;
            e.tl = e.tl.filter(function(x) { return !_cdSameLine(x, ref); });
            if (e.tl.length === n) return false;
            _cdPrune(store, sym);
            return true;
        });
    }
    // An anchor drag finished: move the saved copy (if there is one) from the OLD anchors to the new ones.
    function _cdMoveTl(sym, oldRef, newL, newR) {
        var a = _cdAnch(newL), b = _cdAnch(newR);
        if (!sym || !oldRef || !oldRef.l || !oldRef.r || !a || !b) return;
        var moved = a.time <= b.time ? { l: a, r: b } : { l: b, r: a };
        if (_cdSameLine(moved, oldRef)) return;                  // grabbed an anchor but didn't move it
        _cdMutate(function(store) {
            var e = store[sym];
            if (!e || !e.tl) return false;
            var i = -1;
            for (var k = 0; k < e.tl.length; k++) { if (_cdSameLine(e.tl[k], oldRef)) { i = k; break; } }
            if (i < 0) return false;                             // not a saved line (e.g. alert-backed only)
            moved.dotted = !!e.tl[i].dotted;
            e.tl.splice(i, 1);
            if (!e.tl.some(function(x) { return _cdSameLine(x, moved); })) e.tl.push(moved);
            return true;
        });
    }
    // An AVWAP was placed at bar `idx` of `ohlcv`.
    function _cdAddAv(sym, ohlcv, tf, idx) {
        var bar = ohlcv && ohlcv[idx];
        var u = bar ? _cdNum(bar.time) : null;
        if (!sym || u == null) return;
        _cdMutate(function(store) {
            var e = _cdEntry(store, sym);
            if (e.av.some(function(x) { return _chartAnchorIdx(ohlcv, tf, x) === idx; })) { _cdPrune(store, sym); return false; }
            e.av.push(u);
            return true;
        });
    }
    // An AVWAP at bar `idx` was deleted. Drops every saved anchor that RESOLVES to that bar on this chart -- exactly
    // what restore would have drawn there -- so it still works when the line was placed on another timeframe.
    function _cdDropAv(sym, ohlcv, tf, idx) {
        if (!sym || idx == null) return;
        _cdMutate(function(store) {
            var e = store[sym];
            if (!e || !e.av) return false;
            var n = e.av.length;
            e.av = e.av.filter(function(x) { return _chartAnchorIdx(ohlcv, tf, x) !== idx; });
            if (e.av.length === n) return false;
            _cdPrune(store, sym);
            return true;
        });
    }

    // ── Saved measurements ─────────────────────────────────────────────────────────────────────────────
    // A committed measurement is kept per ticker in the same store, by its two anchors (absolute time + price) in the
    // order they were drawn: the order is the direction (up / down), so (a, b) and (b, a) are different measurements.
    // Like the trendlines they therefore don't depend on the timeframe. Each on-screen item carries `.sym`, the ticker
    // it was drawn on / restored for, so deleting it always drops the right ticker's saved copy. Clearing the screen on
    // a ticker / timeframe change (_measureClearAll) never touches the store.
    function _cdMsRec(m) {
        var a = m && _cdAnch({ time: m.startTime, price: m.startPrice });
        var b = m && _cdAnch({ time: m.endTime,   price: m.endPrice });
        return (a && b) ? { a: a, b: b } : null;
    }
    function _cdSameMs(x, y) { return _cdSamePt(x.a, y.a) && _cdSamePt(x.b, y.b); }
    // A measurement was committed (second click).
    function _cdAddMs(sym, m) {
        var rec = _cdMsRec(m);
        if (!sym || !rec) return;
        _cdMutate(function(store) {
            var e = _cdEntry(store, sym);
            if (e.ms.some(function(x) { return _cdSameMs(x, rec); })) { _cdPrune(store, sym); return false; }
            e.ms.push(rec);
            return true;
        });
    }
    // A measurement was deleted.
    function _cdDropMs(sym, m) {
        var rec = _cdMsRec(m);
        if (!sym || !rec) return;
        _cdMutate(function(store) {
            var e = store[sym];
            if (!e || !e.ms) return false;
            var n = e.ms.length;
            e.ms = e.ms.filter(function(x) { return !_cdSameMs(x, rec); });
            if (e.ms.length === n) return false;
            _cdPrune(store, sym);
            return true;
        });
    }
    // Draw this ticker's saved measurements onto a chart that was just built (fullscreen, watchlist and alerts charts).
    // Kept apart from _cdRestore because the alerts chart restores measurements but not hand-drawn trendlines / AVWAPs.
    // cfg: { isStale(), getOhlcv(), getMeasures(), addMeasure(a, b), render() }
    function _cdRestoreMs(sym, cfg) {
        _cdLoad().then(function(store) {
            if (cfg.isStale()) return;                           // the chart was rebuilt / closed while KV was answering
            var e = store[sym];
            if (!e || !e.ms || !e.ms.length) return;
            var ohlcv = cfg.getOhlcv();
            if (!ohlcv || !ohlcv.length) return;
            var added = false;
            e.ms.forEach(function(s) {
                if (!s || !s.a || !s.b) return;
                var have = cfg.getMeasures().some(function(m) {
                    var r = _cdMsRec(m);
                    return !!r && _cdSameMs(r, s);
                });
                if (have) return;
                cfg.addMeasure(s.a, s.b);
                added = true;
            });
            if (added) cfg.render();
        });
    }

    // Draw this ticker's saved lines onto a chart that was just built. Call it AFTER _restoreAlertLines so an
    // alert-backed copy of the same line is already there and is not drawn twice.
    // cfg: { isStale(), getOhlcv(), getTrendlines(), getVwapAnchors(), addTrendline(p1,p2,extend,dotted), addVwap(idx) }
    function _cdRestore(sym, tf, cfg) {
        _cdLoad().then(function(store) {
            if (cfg.isStale()) return;                           // the chart was rebuilt / closed while KV was answering
            var e = store[sym];
            if (!e) return;
            var ohlcv = cfg.getOhlcv();
            if (!ohlcv || !ohlcv.length) return;
            (e.tl || []).forEach(function(s) {
                if (!s || !s.l || !s.r) return;
                var have = cfg.getTrendlines().some(function(t) {
                    return t.leftP && t.rightP && _cdSameLine({ l: t.leftP, r: t.rightP }, s);
                });
                if (have) return;
                cfg.addTrendline({ time: s.l.time, price: s.l.price }, { time: s.r.time, price: s.r.price }, false, !!s.dotted);
            });
            (e.av || []).forEach(function(u) {
                var idx = _chartAnchorIdx(ohlcv, tf, u);
                if (idx < 0 || cfg.getVwapAnchors().indexOf(idx) !== -1) return;
                cfg.addVwap(idx);
            });
        });
    }

    // ── Shared hit-test / selection cores — used by fullscreen, watchlist, and
    // alerts charts (alerts.js calls these cross-file, same pattern as fetchMcOhlcv).
    function _trendlineHitTestCore(clientX, clientY, chart, candle, ohlcv, trendlines, contRef) {
        if (!chart || !candle || !trendlines.length || !contRef) return -1;
        var rect     = contRef.getBoundingClientRect();
        var px       = clientX - rect.left;
        var py       = clientY - rect.top;
        var HIT_PX   = 7;
        var bestIdx  = -1;
        var bestDist = HIT_PX;
        trendlines.forEach(function(tl, idx) {
            var x1 = _mcFsTimeToX(chart, ohlcv, tl.leftP.time);
            var x2 = _mcFsTimeToX(chart, ohlcv, tl.rightP.time);
            var y1 = candle.priceToCoordinate(tl.leftP.price);
            var y2 = candle.priceToCoordinate(tl.rightP.price);
            if (x1 == null || x2 == null || y1 == null || y2 == null) return;
            var dx = x2 - x1, dy = y2 - y1;
            var lenSq = dx * dx + dy * dy;
            var dist;
            if (lenSq === 0) {
                dist = Math.hypot(px - x1, py - y1);
            } else {
                var t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
                dist  = Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
            }
            if (dist < bestDist) { bestDist = dist; bestIdx = idx; }
        });
        return bestIdx;
    }

    function _deselectAllTrendlinesCore(trendlines) {
        trendlines.forEach(function(tl) {
            if (tl.selected) { tl.selected = false; if (tl.requestUpdate) tl.requestUpdate(); }
        });
    }

    // A selected AVWAP is marked with fixed dots along its line -- the same dot a selected trendline shows on each of its
    // anchors -- instead of a thicker line. Dots sit on the anchor bar, then every _VWAP_DOT_EVERY_BARS bars counted from
    // the anchor, plus one on the line's last bar (a regular dot that would land within half a step of the last-bar dot
    // is dropped so the two don't crowd). The count depends only on how many bars the line spans -- never on zoom -- and
    // the dots stay pinned to the same bars while panning. Only on-screen dots are drawn. The dots are a canvas
    // primitive on the AVWAP's own series (the primitive API the trendlines already use), created the first time that
    // AVWAP is selected and then just shown / hidden.
    // If it can't be attached or has no anchor point, the selected AVWAP is drawn DASHED instead (never thicker), so a
    // selection is never invisible.
    var _VWAP_DOT_EVERY_BARS = 25;   // bars between selection dots (50-bar line -> 3 dots: start, middle, end)
    var _VWAP_DOT_RADIUS = 4.5;      // same radius the anchor dot has always had
    function _vwapAnchorPt(entry) {
        try {
            var it = entry && entry.dataMap && entry.dataMap.entries().next();   // first point = the anchor bar
            if (it && !it.done) return { time: it.value[0], value: it.value[1] };
        } catch (e) {}
        return null;
    }
    function _makeVwapAnchorDot(entry) {
        var chartRef = null, seriesRef = null, requestUpdate = null;
        var prim = {
            shown: false,
            attached: function(param) { chartRef = param.chart; seriesRef = param.series; requestUpdate = param.requestUpdate; },
            detached: function() { chartRef = seriesRef = requestUpdate = null; },
            show: function(on) { prim.shown = !!on; if (requestUpdate) { try { requestUpdate(); } catch (e) {} } },
            paneViews: function() {
                return [{
                    zOrder: function() { return 'top'; },
                    renderer: function() {
                        return {
                            draw: function(target) {
                                if (!prim.shown || !chartRef || !seriesRef || !entry || !entry.dataMap) return;
                                var ts = chartRef.timeScale();
                                // Which bars (counted from the anchor) get a dot: anchor, every Nth, and the last bar.
                                var step = _VWAP_DOT_EVERY_BARS;
                                var lastIdx = entry.dataMap.size - 1;
                                var lastRegular = Math.floor(lastIdx / step) * step;
                                // A regular dot within half a step of the last bar is dropped (never the anchor itself)
                                var dropIdx = (lastRegular > 0 && lastRegular < lastIdx && (lastIdx - lastRegular) < step / 2) ? lastRegular : -1;
                                target.useBitmapCoordinateSpace(function(scope) {
                                    var ctx = scope.context, rx = scope.horizontalPixelRatio, ry = scope.verticalPixelRatio;
                                    var w = scope.mediaSize ? scope.mediaSize.width : Infinity;
                                    var h = scope.mediaSize ? scope.mediaSize.height : Infinity;
                                    var r = _VWAP_DOT_RADIUS;
                                    ctx.save();
                                    // Same look as the trendline anchor handles (see _addTrendlineCore)
                                    ctx.fillStyle   = themeColor('text-emphasis');
                                    ctx.strokeStyle = themeColor('bg-page');
                                    ctx.lineWidth   = 1.5 * rx;
                                    var i = 0;
                                    // dataMap is in time order, first entry = anchor bar, last entry = the line's last bar
                                    entry.dataMap.forEach(function(value, time) {
                                        var idx = i++;
                                        var wanted = (idx === 0) || (idx === lastIdx) || (idx % step === 0 && idx !== dropIdx);
                                        if (!wanted) return;
                                        var x = ts.timeToCoordinate(time);
                                        if (x == null || x < -r || x > w + r) return;
                                        var y = seriesRef.priceToCoordinate(value);
                                        if (y == null || y < -r || y > h + r) return;
                                        ctx.beginPath();
                                        ctx.arc(x * rx, y * ry, r * rx, 0, Math.PI * 2);
                                        ctx.fill();
                                        ctx.stroke();
                                    });
                                    ctx.restore();
                                });
                            }
                        };
                    }
                }];
            }
        };
        return prim;
    }
    function _vwapSetSelectedLook(entry, selected) {
        if (selected && !entry._dot && !entry._dotFailed) {
            try {
                if (typeof entry.series.attachPrimitive !== 'function' || !_vwapAnchorPt(entry)) throw new Error('no dot');
                var d = _makeVwapAnchorDot(entry);
                entry.series.attachPrimitive(d);
                entry._dot = d;
            } catch (e) { entry._dotFailed = true; entry._dot = null; }
        }
        if (entry._dot) entry._dot.show(selected);
        // lineStyle 0 = solid, 2 = dashed (the library's numeric values). Width never changes.
        entry.series.applyOptions({ lineWidth: 1.5, lineStyle: (selected && !entry._dot) ? 2 : 0 });
    }

    function _selectVwapCore(vwapSeries, idx) {
        vwapSeries.forEach(function(entry, i) { _vwapSetSelectedLook(entry, i === idx); });
    }

    function _deselectAllVwapsCore(vwapSeries) {
        vwapSeries.forEach(function(entry) { _vwapSetSelectedLook(entry, false); });
    }

    function _vwapHitTestCore(clientX, clientY, chart, vwapSeries, lastCrosshairTime, containerId) {
        if (!chart || !vwapSeries.length || !lastCrosshairTime) return -1;
        var chartDiv = document.getElementById(containerId);
        var rect = chartDiv ? chartDiv.getBoundingClientRect() : null;
        if (!rect) return -1;
        var localY   = clientY - rect.top;
        var HIT_PX   = 8;
        var bestDist = HIT_PX;
        var hitIdx   = -1;
        vwapSeries.forEach(function(entry, i) {
            if (!entry.dataMap) return;
            var avwapVal = entry.dataMap.get(lastCrosshairTime);
            if (avwapVal == null) return;
            var yCoord = entry.series.priceToCoordinate(avwapVal);
            if (yCoord == null) return;
            var dist = Math.abs(localY - yCoord);
            if (dist < bestDist) { bestDist = dist; hitIdx = i; }
        });
        return hitIdx;
    }

    // Returns 'left' or 'right' if clientX/Y is near an anchor of trendlines[tlIdx], else null
    function _anchorHitTestCore(clientX, clientY, tlIdx, trendlines, chart, candle, ohlcv, contRef) {
        if (tlIdx < 0 || !trendlines[tlIdx] || !chart || !candle || !contRef) return null;
        var tl   = trendlines[tlIdx];
        var rect = contRef.getBoundingClientRect();
        var px   = clientX - rect.left;
        var py   = clientY - rect.top;
        var HIT  = 10;
        var x1 = _mcFsTimeToX(chart, ohlcv, tl.leftP.time);
        var y1 = candle.priceToCoordinate(tl.leftP.price);
        if (x1 != null && y1 != null && Math.hypot(px - x1, py - y1) <= HIT) return 'left';
        var x2 = _mcFsTimeToX(chart, ohlcv, tl.rightP.time);
        var y2 = candle.priceToCoordinate(tl.rightP.price);
        if (x2 != null && y2 != null && Math.hypot(px - x2, py - y2) <= HIT) return 'right';
        return null;
    }

    // ── Trendline hit-test: returns _mcFsTrendlines index within HIT_PX, or -1
    function _trendlineHitTest(clientX, clientY) {
        return _trendlineHitTestCore(clientX, clientY, _mcFsChart, _mcFsCandle, _mcFsOhlcv, _mcFsTrendlines, _mcFsTrendContRef);
    }

    // ── Selection helpers ─────────────────────────────────────────────────────
    function _deselectAllTrendlines() {
        _deselectAllTrendlinesCore(_mcFsTrendlines);
        _mcFsSelectedTrendlineIdx = -1;
    }

    function _selectVwap(idx) {
        _selectVwapCore(_mcFsVwapSeries, idx);
        _mcFsSelectedVwapIdx = idx;
    }
    function _deselectAllVwaps() {
        _deselectAllVwapsCore(_mcFsVwapSeries);
        _mcFsSelectedVwapIdx = -1;
    }

    function _mcFsVwapHitTest(clientX, clientY) {
        return _vwapHitTestCore(clientX, clientY, _mcFsChart, _mcFsVwapSeries, _mcFsLastCrosshairTime, 'mc-fullscreen-chart');
    }

    function _anchorHitTest(clientX, clientY, tlIdx) {
        return _anchorHitTestCore(clientX, clientY, tlIdx, _mcFsTrendlines, _mcFsChart, _mcFsCandle, _mcFsOhlcv, _mcFsTrendContRef);
    }

    // ── Anchor drag ───────────────────────────────────────────────────────────
    // Shared anchor-drag core — used by fullscreen/watchlist/alerts trendline dragging
    function _onTrendAnchorDragMoveCore(evt, cfg) {
        var dragState = cfg.dragState;
        if (!dragState || !cfg.chart || !cfg.candle || !cfg.contRef) return;
        var tl = cfg.trendlines[dragState.tlIdx];
        if (!tl) return;
        cfg.contRef.style.cursor = 'grabbing';
        var rect  = cfg.contRef.getBoundingClientRect();
        var lx    = evt.clientX - rect.left;
        var ly    = evt.clientY - rect.top;
        var price = cfg.candle.coordinateToPrice(ly);
        var time  = cfg.chart.timeScale().coordinateToTime(lx);
        if (price == null) return;
        // Allow dragging into the future: if coordinateToTime returns null (off right edge),
        // extrapolate from the last bar interval so the anchor can be placed in future space
        if (time == null) {
            var ohlcv  = cfg.ohlcv;
            var last   = ohlcv[ohlcv.length - 1];
            var prev   = ohlcv[ohlcv.length - 2] || last;
            var barSec = ohlcv.length >= 2 ? (last.time - prev.time) : 86400;
            var lastX  = cfg.chart.timeScale().timeToCoordinate(last.time);
            if (lastX == null) return;
            var prevX    = cfg.chart.timeScale().timeToCoordinate(prev.time);
            var pxPerBar = prevX != null ? Math.abs(lastX - prevX) : 8;
            var barsAhead = pxPerBar > 0 ? Math.max(1, Math.round((lx - lastX) / pxPerBar)) : 1;
            time = _mcFutureTime(ohlcv, barsAhead);
        }
        var newAnchor = { time: time, price: price };
        if (dragState.anchorSide === 'left') {
            tl.leftP = newAnchor;
        } else {
            tl.rightP = newAnchor;
        }
        // Re-normalise if anchors have crossed (compare by time)
        if (tl.leftP.time > tl.rightP.time) {
            var tmp = tl.leftP; tl.leftP = tl.rightP; tl.rightP = tmp;
            dragState.anchorSide = dragState.anchorSide === 'left' ? 'right' : 'left';
        }
        tl.p1 = tl.leftP; tl.p2 = tl.rightP;
        // Drive the SVG preview instantly (no LW canvas re-render on every move)
        if (cfg.svgOverlay && cfg.svgLine && dragState.fixedX != null) {
            cfg.svgLine.setAttribute('x2', lx);
            cfg.svgLine.setAttribute('y2', ly);
        }
    }

    function _onTrendAnchorDragEndCore(cfg) {
        var state = cfg.getDragState();
        cfg.setDragState(null);
        document.removeEventListener('mousemove', cfg.moveHandler);
        document.removeEventListener('mouseup',   cfg.endHandler);
        if (cfg.contRef) cfg.contRef.style.cursor = '';
        // Re-enable canvas draw and commit final position
        if (state) {
            var tl = cfg.trendlines[state.tlIdx];
            if (tl) {
                tl.dragging = false; if (tl.requestUpdate) tl.requestUpdate();
                // Dragging used to move only the drawing: the alert kept monitoring the OLD line (and the old line
                // was redrawn from the alert store on the next open). If this drawing backs an alert, move it too.
                if (cfg.getSym && tl.alertRef && tl.leftP && tl.rightP && window.alSyncTrendlineAlertsAfterDrag) {
                    try { window.alSyncTrendlineAlertsAfterDrag(cfg.getSym(), tl.alertRef.l, tl.alertRef.r, tl.leftP, tl.rightP); } catch (e) {}
                }
                // Move the SAVED copy of this line too (see "Saved chart drawings"). Must run before alertRef is
                // overwritten below: the old anchors are how the saved entry is found. No-op if the line isn't saved
                // (e.g. an alert-backed line), and harmless on the alerts chart, which shares this core.
                if (cfg.getSym && tl.alertRef && tl.leftP && tl.rightP) {
                    try { _cdMoveTl(cfg.getSym(), tl.alertRef, tl.leftP, tl.rightP); } catch (e) {}
                }
                if (tl.leftP && tl.rightP) {
                    tl.alertRef = { l: { time: tl.leftP.time, price: tl.leftP.price }, r: { time: tl.rightP.time, price: tl.rightP.price } };
                }
            }
        }
        // Hide SVG after two rAFs so LW canvas has time to paint the committed line
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                if (cfg.svgOverlay) cfg.svgOverlay.style.display = 'none';
            });
        });
    }

    function _onTrendAnchorDragMove(evt) {
        _onTrendAnchorDragMoveCore(evt, {
            dragState:  _mcFsTrendDragState,
            chart:      _mcFsChart,
            candle:     _mcFsCandle,
            contRef:    _mcFsTrendContRef,
            trendlines: _mcFsTrendlines,
            ohlcv:      _mcFsOhlcv,
            svgOverlay: _mcFsTrendSvgOverlay,
            svgLine:    _mcFsTrendSvgLine
        });
    }

    function _onTrendAnchorDragEnd() {
        _onTrendAnchorDragEndCore({
            getDragState: function() { return _mcFsTrendDragState; },
            setDragState: function(v) { _mcFsTrendDragState = v; },
            getSym:       function() { return _mcFsSym; },
            trendlines:   _mcFsTrendlines,
            contRef:      _mcFsTrendContRef,
            svgOverlay:   _mcFsTrendSvgOverlay,
            moveHandler:  _onTrendAnchorDragMove,
            endHandler:   _onTrendAnchorDragEnd
        });
    }

    // ── Measure-tool drag core — used by fullscreen/watchlist/alerts ─────────
    function _onMeasureDragMoveCore(evt, cfg) {
        if (!cfg.getActive() || !cfg.contRef || !cfg.chart || !cfg.candle) return;
        if (cfg.getRafId()) return; // already a frame queued — skip raw event
        var cx = evt.clientX, cy = evt.clientY;
        cfg.setRafId(requestAnimationFrame(function() {
            cfg.setRafId(null);
            if (!cfg.getActive()) return;
            var r  = cfg.contRef.getBoundingClientRect();
            var lx = cx - r.left;
            var ly = cy - r.top;
            var eP = cfg.candle.coordinateToPrice(ly);
            var eT = _measureGetTimeAtX(cfg.chart, cfg.ohlcv, lx);
            if (eP == null || eT == null) return;
            var result = _computeMeasureResult(cfg.ohlcv, cfg.getStart().time, cfg.getStart().price, eT, eP);
            cfg.setResult(result);
            _renderMeasureOverlay(cfg.chart, cfg.candle, cfg.contRef,
                cfg.svgOverlay, cfg.svgRect, cfg.hLine, cfg.infoDiv, result);
        }));
    }
    function _onMeasureDragEndCore(cfg) {
        document.removeEventListener('mousemove', cfg.moveHandler);
        document.removeEventListener('mouseup',   cfg.endHandler);
        cfg.setActive(false);
        // Result stays visible; cleared on next non-measure click or Escape
    }
    // Two-click preview: fires on free mousemove after first Shift+click (no button held)
    function _onMeasurePreviewMoveCore(evt, cfg) {
        if (!cfg.getActive() || cfg.getPhase() !== 1 || !cfg.contRef || !cfg.chart || !cfg.candle) return;
        if (cfg.getRafId()) return;
        var cx = evt.clientX, cy = evt.clientY;
        cfg.setRafId(requestAnimationFrame(function() {
            cfg.setRafId(null);
            if (!cfg.getActive() || cfg.getPhase() !== 1) return;
            var r  = cfg.contRef.getBoundingClientRect();
            var lx = cx - r.left;
            var ly = cy - r.top;
            var eP = cfg.candle.coordinateToPrice(ly);
            var eT = _measureGetTimeAtX(cfg.chart, cfg.ohlcv, lx);
            if (eP == null || eT == null) return;
            var result = _computeMeasureResult(cfg.ohlcv, cfg.getStart().time, cfg.getStart().price, eT, eP);
            cfg.setResult(result);
            _renderMeasureOverlay(cfg.chart, cfg.candle, cfg.contRef,
                cfg.svgOverlay, cfg.svgRect, cfg.hLine, cfg.infoDiv, result);
        }));
    }

    // Abandon a half-finished measurement (first click placed, second never came) so its anchor
    // and the live mousemove listener can't leak onto the next symbol or the next open, and drop the
    // committed measurements (same lifetime as the trendlines).
    function _mcFsResetMeasure() {
        _mcFsMeasureActive = false; _mcFsMeasurePhase = 0; _mcFsMeasureStart = null; _mcFsMeasureResult = null;
        _measureClearAll(_mcFsMeasureList);
        if (_mcFsMeasureRafId) { cancelAnimationFrame(_mcFsMeasureRafId); _mcFsMeasureRafId = null; }
        document.removeEventListener('mousemove', _onMcFsMeasurePreviewMove);
        _hideMeasureOverlay(_mcFsMeasureSvgOverlay, _mcFsMeasureInfoDiv);
    }

    // ── mc-fs Measure drag handlers ──────────────────────────────────────────
    function _onMcFsMeasureDragMove(evt) {
        _onMeasureDragMoveCore(evt, {
            getActive:  function() { return _mcFsMeasureActive; },
            contRef:    _mcFsTrendContRef,
            chart:      _mcFsChart,
            candle:     _mcFsCandle,
            ohlcv:      _mcFsOhlcv,
            getStart:   function() { return _mcFsMeasureStart; },
            getRafId:   function() { return _mcFsMeasureRafId; },
            setRafId:   function(v) { _mcFsMeasureRafId = v; },
            setResult:  function(v) { _mcFsMeasureResult = v; },
            svgOverlay: _mcFsMeasureSvgOverlay,
            svgRect:    _mcFsMeasureSvgRect,
            hLine:      _mcFsMeasureHLine,
            infoDiv:    _mcFsMeasureInfoDiv
        });
    }
    function _onMcFsMeasureDragEnd() {
        _onMeasureDragEndCore({
            moveHandler: _onMcFsMeasureDragMove,
            endHandler:  _onMcFsMeasureDragEnd,
            setActive:   function(v) { _mcFsMeasureActive = v; }
        });
    }
    function _onMcFsMeasurePreviewMove(evt) {
        _onMeasurePreviewMoveCore(evt, {
            getActive:  function() { return _mcFsMeasureActive; },
            getPhase:   function() { return _mcFsMeasurePhase; },
            contRef:    _mcFsTrendContRef,
            chart:      _mcFsChart,
            candle:     _mcFsCandle,
            ohlcv:      _mcFsOhlcv,
            getStart:   function() { return _mcFsMeasureStart; },
            getRafId:   function() { return _mcFsMeasureRafId; },
            setRafId:   function(v) { _mcFsMeasureRafId = v; },
            setResult:  function(v) { _mcFsMeasureResult = v; },
            svgOverlay: _mcFsMeasureSvgOverlay,
            svgRect:    _mcFsMeasureSvgRect,
            hLine:      _mcFsMeasureHLine,
            infoDiv:    _mcFsMeasureInfoDiv
        });
    }

    // ── Trendline: click → free-move preview → click-to-finish ────────────
    // First click sets the start anchor; mouse movement (no button held) shows a
    // live extended-line preview; second click finalises.  Uses raw DOM mousedown
    // in capture phase so LW Charts never sees the event and cannot start a pan.

    // The SVG preview line (shown while drawing and while dragging an anchor) must match the style of the line it
    // previews, otherwise a dotted line would flip to solid for the duration of a drag. Called every time the
    // preview is shown, so it also clears the dots when the next line is solid.
    function _applyTrendlineDash(svgLine, dotted) {
        if (!svgLine) return;
        if (dotted) { svgLine.setAttribute('stroke-dasharray', '0.1 5.9'); svgLine.setAttribute('stroke-linecap', 'round'); svgLine.setAttribute('stroke-width', '2.5'); }
        else        { svgLine.removeAttribute('stroke-dasharray');         svgLine.removeAttribute('stroke-linecap');         svgLine.setAttribute('stroke-width', '1.5'); } // 1.5 = the width every chart's preview line is created with
    }

    // ── Shared trendline mousedown core — used by all three charts. Handles the
    // measure-tool intercept, anchor-drag pickup, line select/deselect, and the
    // two-click draw flow. (alerts.js calls this cross-file.)
    function _onTrendMouseDownCore(evt, cfg) {
        if (evt.button !== 0 || !cfg.candle || !cfg.chart || !cfg.contRef) return;

        // ── Adjust a placed measurement: grab an end handle of the SELECTED one ──────────────────────────
        // Checked before the measure-tool intercept so it works with the tool on or off, but never while a new
        // measurement is half-placed (that click must finish it) or a trendline is being drawn / dragged.
        if (cfg.measureList && cfg.measureList.length && cfg.getMeasurePhase() !== 1 && !cfg.getDragState() &&
            !cfg.getTrendlineMode() && !cfg.trendDraw.active) {
            var _mhHit = _measureHandleHit(cfg.measureList, cfg.contRef, evt.clientX, evt.clientY);
            if (_mhHit) {
                evt.stopPropagation();
                evt.preventDefault();
                _measureBeginDrag(cfg, _mhHit.m, _mhHit.which, evt);
                return;
            }
        }

        // ── Measure tool intercept ───────────────────────────────────────────
        if ((evt.shiftKey || cfg.getMeasureMode()) && !cfg.getDragState()) {
            evt.stopPropagation();
            evt.preventDefault();
            if (cfg.trendDraw.active) {
                cfg.trendDraw.active = false; cfg.trendDraw.startTime = null; cfg.trendDraw.startPrice = null;
                if (cfg.svgOverlay) cfg.svgOverlay.style.display = 'none';
            }
            var _mRect = cfg.contRef.getBoundingClientRect();
            var _mlx   = evt.clientX - _mRect.left;
            var _mly   = evt.clientY - _mRect.top;
            var _mP    = cfg.candle.coordinateToPrice(_mly);
            var _mT    = _measureGetTimeAtX(cfg.chart, cfg.ohlcv, _mlx);
            if (_mP == null || _mT == null) return;
            // The click that PLACES an anchor (first or second) snaps to the closest wick tip (a high or a low of
            // the candles around the cursor, by on-screen distance), moving the anchor to that tip's candle and price.
            // The live preview between the clicks stays free.
            var _mTip = _measureSnapTip(cfg.chart, cfg.candle, cfg.ohlcv, _mlx, _mly);
            if (_mTip) { _mT = _mTip.time; _mP = _mTip.price; }
            var _mSi   = _barIdxByTime(cfg.ohlcv, _mT);

            if (cfg.getMeasurePhase() === 1) {
                // Second click — finalise at current cursor position
                // Commit: the measurement becomes a persistent item (stays until deleted) and the live preview is hidden.
                // A second click on the exact anchor point (double-click) would only leave a zero-size box, so it is dropped.
                var _msA = cfg.getMeasureStart();
                if (_msA.time !== _mT || _msA.price !== _mP) {
                    var _msItem = _measureCommit(cfg, _msA.time, _msA.price, _mT, _mP);
                    _measureRenderAll(cfg.measureList, cfg.chart, cfg.candle, cfg.contRef, cfg.ohlcv);
                    if (cfg.onMeasureCommitted) { try { cfg.onMeasureCommitted(_msItem); } catch (e) {} }   // saved per ticker (_cdAddMs)
                }
                cfg.setMeasureResult(null);
                _hideMeasureOverlay(cfg.measureSvgOverlay, cfg.measureInfoDiv);
                cfg.setMeasureActive(false);
                cfg.setMeasurePhase(0);
                if (cfg.getMeasureRafId()) { cancelAnimationFrame(cfg.getMeasureRafId()); cfg.setMeasureRafId(null); }
                document.removeEventListener('mousemove', cfg.measurePreviewMoveHandler);
                return;
            }

            // First click — set anchor, enter free-move preview phase
            cfg.setMeasureStart({ time: _mT, price: _mP, barIdx: _mSi });
            cfg.setMeasureResult(null);
            cfg.setMeasureActive(true);
            cfg.setMeasurePhase(1);
            _hideMeasureOverlay(cfg.measureSvgOverlay, cfg.measureInfoDiv);
            document.removeEventListener('mousemove', cfg.measurePreviewMoveHandler); // clear stale
            document.addEventListener('mousemove', cfg.measurePreviewMoveHandler);
            return;
        }

        // Plain click (no shift, no measure mode) — cancel phase-1 preview or clear result
        if (cfg.getMeasurePhase() === 1) {
            cfg.setMeasureActive(false);
            cfg.setMeasurePhase(0);
            if (cfg.getMeasureRafId()) { cancelAnimationFrame(cfg.getMeasureRafId()); cfg.setMeasureRafId(null); }
            document.removeEventListener('mousemove', cfg.measurePreviewMoveHandler);
            _hideMeasureOverlay(cfg.measureSvgOverlay, cfg.measureInfoDiv);
            cfg.setMeasureResult(null);
        } else if (cfg.getMeasureResult() && !cfg.getMeasureMode()) {
            _hideMeasureOverlay(cfg.measureSvgOverlay, cfg.measureInfoDiv);
            cfg.setMeasureResult(null);
        }

        // Click on a committed measurement (outline or label) selects it; Delete then removes it. Any other
        // click deselects. Not while drawing a trendline, so its anchor clicks can't be swallowed.
        if (cfg.measureList && cfg.measureList.length && !cfg.getTrendlineMode() && !cfg.trendDraw.active) {
            var _msHit = _measureHitTest(cfg.measureList, cfg.contRef, evt.clientX, evt.clientY);
            if (_msHit) {
                evt.stopPropagation();
                evt.preventDefault();
                cfg.deselectAllTrendlines();
                cfg.deselectAllVwaps();
                _measureSelect(cfg.measureList, _msHit);
                _measureBeginDrag(cfg, _msHit, 'move', evt);   // drag from here to move the whole measurement
                return;
            }
            _measureSelect(cfg.measureList, null);
        }

        // ── Phase 1: anchor drag — check selected line first, then all others
        if (!cfg.trendDraw.active) {
            var dragTlIdx = -1, anchorSide = null;
            // Prefer the already-selected line so its anchors take priority
            var selIdx = cfg.getSelectedIdx();
            if (selIdx !== -1) {
                anchorSide = cfg.anchorHitTest(evt.clientX, evt.clientY, selIdx);
                if (anchorSide) dragTlIdx = selIdx;
            }
            // Fall back to any other line's anchors
            if (dragTlIdx === -1) {
                for (var _di = 0; _di < cfg.trendlines.length; _di++) {
                    var _as = cfg.anchorHitTest(evt.clientX, evt.clientY, _di);
                    if (_as) { dragTlIdx = _di; anchorSide = _as; break; }
                }
            }
            if (dragTlIdx !== -1) {
                evt.stopPropagation();
                // Auto-select the line if it wasn't already selected
                if (cfg.getSelectedIdx() !== dragTlIdx) {
                    cfg.deselectAllTrendlines();
                    cfg.setSelectedIdx(dragTlIdx);
                    cfg.trendlines[dragTlIdx].selected = true;
                    if (cfg.trendlines[dragTlIdx].requestUpdate) cfg.trendlines[dragTlIdx].requestUpdate();
                }
                var _dragTl   = cfg.trendlines[dragTlIdx];
                var _fixedP   = anchorSide === 'left' ? _dragTl.rightP : _dragTl.leftP;
                // Use _mcFsTimeToX so future-anchored fixed points also resolve correctly
                var _fixedX   = _mcFsTimeToX(cfg.chart, cfg.ohlcv, _fixedP.time);
                var _fixedY   = cfg.candle.priceToCoordinate(_fixedP.price);
                cfg.setDragState({ tlIdx: dragTlIdx, anchorSide: anchorSide, fixedX: _fixedX, fixedY: _fixedY });
                // Suppress the canvas line so only the SVG preview is visible during drag
                _dragTl.dragging = true;
                if (_dragTl.requestUpdate) _dragTl.requestUpdate();
                // Kick off SVG drag-preview immediately (same overlay used while drawing)
                if (cfg.svgOverlay && cfg.svgLine && _fixedX != null && _fixedY != null) {
                    var _dRect = cfg.contRef.getBoundingClientRect();
                    var _curX  = evt.clientX - _dRect.left;
                    var _curY  = evt.clientY - _dRect.top;
                    cfg.svgLine.setAttribute('x1', _fixedX); cfg.svgLine.setAttribute('y1', _fixedY);
                    cfg.svgLine.setAttribute('x2', _curX);   cfg.svgLine.setAttribute('y2', _curY);
                    // Re-resolve the colour every time the preview is shown — the overlay is created once
                    // and reused, so a stroke set at creation goes stale after a light/dark toggle.
                    cfg.svgLine.setAttribute('stroke', _TRENDLINE_COLOR());
                    _applyTrendlineDash(cfg.svgLine, !!_dragTl.dotted);
                    cfg.svgOverlay.style.display = '';
                }
                document.addEventListener('mousemove', cfg.dragMoveHandler);
                document.addEventListener('mouseup',   cfg.dragEndHandler);
                return;
            }
        }

        // ── Phase 2: hit-test lines (select / deselect)
        if (!cfg.trendDraw.active) {
            var hitIdx = cfg.trendlineHitTest(evt.clientX, evt.clientY);
            if (hitIdx !== -1) {
                evt.stopPropagation();
                var curSel = cfg.getSelectedIdx();
                if (curSel !== -1 && curSel !== hitIdx) {
                    var prev = cfg.trendlines[curSel];
                    if (prev) { prev.selected = false; if (prev.requestUpdate) prev.requestUpdate(); }
                }
                cfg.setSelectedIdx(hitIdx);
                cfg.trendlines[hitIdx].selected = true;
                if (cfg.trendlines[hitIdx].requestUpdate) cfg.trendlines[hitIdx].requestUpdate();
                return;
            }
            // No hit — deselect
            if (cfg.getSelectedIdx() !== -1) cfg.deselectAllTrendlines();
        }

        // ── Phase 3: drawing mode guard
        if (!cfg.getTrendlineMode()) return;
        evt.stopPropagation();

        var rect  = cfg.contRef.getBoundingClientRect();
        var lx    = evt.clientX - rect.left;
        var ly    = evt.clientY - rect.top;
        var price = cfg.candle.coordinateToPrice(ly);
        // Within the bar area, use the crosshair time (reliable, snaps to nearest bar).
        // Only switch to pixel extrapolation when the cursor is visually past the last bar.
        var time  = null;
        if (cfg.ohlcv.length >= 2) {
            var _ohlcv   = cfg.ohlcv;
            var _last    = _ohlcv[_ohlcv.length - 1];
            var _prev    = _ohlcv[_ohlcv.length - 2];
            var _lastX   = cfg.chart.timeScale().timeToCoordinate(_last.time);
            var _prevX   = cfg.chart.timeScale().timeToCoordinate(_prev.time);
            var _pxPerBar = (_lastX != null && _prevX != null) ? Math.abs(_lastX - _prevX) : 8;
            if (_lastX != null && lx > _lastX + _pxPerBar * 0.5) {
                // Cursor is past the last bar — extrapolate a future timestamp
                var _barSec   = _last.time - _prev.time;
                var _barsAhead = Math.max(1, Math.round((lx - _lastX) / _pxPerBar));
                time = _mcFutureTime(_ohlcv, _barsAhead);
            } else {
                // Within bar area — crosshair time is reliable
                time = cfg.getLastCrosshairTime() || _last.time;
            }
        }
        if (price == null || time == null) return;
        if (!cfg.trendDraw.active) {
            // ── First click: set start anchor ──────────────────────────────
            cfg.trendDraw.active     = true;
            cfg.trendDraw.startTime  = time;
            cfg.trendDraw.startPrice = price;
            if (cfg.svgOverlay && cfg.svgLine && cfg.chart) {
                var ax = cfg.chart.timeScale().timeToCoordinate(time);
                var ay = cfg.candle.priceToCoordinate(price);
                if (ax != null && ay != null) {
                    cfg.svgLine.setAttribute('x1', ax); cfg.svgLine.setAttribute('y1', ay);
                    cfg.svgLine.setAttribute('x2', ax); cfg.svgLine.setAttribute('y2', ay);
                }
                // Same as the anchor-drag path: refresh the stroke so it matches the current theme.
                cfg.svgLine.setAttribute('stroke', _TRENDLINE_COLOR());
                // Only the fullscreen chart supplies getTrendlineStyle; the watchlist/alerts charts stay solid.
                _applyTrendlineDash(cfg.svgLine, !!(cfg.getTrendlineStyle && cfg.getTrendlineStyle() === 'dotted'));
                cfg.svgOverlay.style.display = '';
            }
        } else {
            // ── Second click: finalise ──────────────────────────────────────
            var p1 = { time: cfg.trendDraw.startTime, price: cfg.trendDraw.startPrice };
            cfg.trendDraw.active = false;
            cfg.trendDraw.startTime = null; cfg.trendDraw.startPrice = null;
            if (cfg.svgOverlay) cfg.svgOverlay.style.display = 'none';
            if (time !== p1.time) {
                var drawnTl = cfg.addTrendline(p1, { time: time, price: price }, false, !!(cfg.getTrendlineStyle && cfg.getTrendlineStyle() === 'dotted'));
                // Opt-in: only the fullscreen and watchlist charts save hand-drawn lines (see "Saved chart drawings").
                if (drawnTl && cfg.onTrendDrawn) cfg.onTrendDrawn(drawnTl);
                // Select the line just drawn so Delete removes it straight away. The draw tool switches itself off
                // below, so the "delete the last line while the tool is on" branch of the key handlers no longer
                // applies. Exclusive with AVWAP selection, because the key handlers check trendlines first.
                var _newTlIdx = drawnTl ? cfg.trendlines.indexOf(drawnTl) : -1;
                if (_newTlIdx !== -1) {
                    cfg.deselectAllTrendlines();
                    if (cfg.deselectAllVwaps) cfg.deselectAllVwaps();
                    cfg.setSelectedIdx(_newTlIdx);
                    drawnTl.selected = true;
                    if (drawnTl.requestUpdate) drawnTl.requestUpdate();
                }
            }
            // Auto-deactivate: turn button off after trendline is drawn
            cfg.setTrendlineMode(false);
            var tDoneBtn = document.getElementById(cfg.doneBtnId);
            if (tDoneBtn) tDoneBtn.classList.remove('active');
        }
    }

    function _onTrendMouseDown(evt) {
        _onTrendMouseDownCore(evt, {
            candle:  _mcFsCandle,
            chart:   _mcFsChart,
            contRef: _mcFsTrendContRef,
            getMeasureMode:    function() { return _mcFsMeasureMode; },
            getDragState:      function() { return _mcFsTrendDragState; },
            setDragState:      function(v) { _mcFsTrendDragState = v; },
            trendDraw:         _mcFsTrendDraw,
            svgOverlay:        _mcFsTrendSvgOverlay,
            svgLine:           _mcFsTrendSvgLine,
            ohlcv:             _mcFsOhlcv,
            getMeasurePhase:   function() { return _mcFsMeasurePhase; },
            setMeasurePhase:   function(v) { _mcFsMeasurePhase = v; },
            getMeasureResult:  function() { return _mcFsMeasureResult; },
            setMeasureResult:  function(v) { _mcFsMeasureResult = v; },
            setMeasureActive:  function(v) { _mcFsMeasureActive = v; },
            getMeasureRafId:   function() { return _mcFsMeasureRafId; },
            setMeasureRafId:   function(v) { _mcFsMeasureRafId = v; },
            getMeasureStart:   function() { return _mcFsMeasureStart; },
            setMeasureStart:   function(v) { _mcFsMeasureStart = v; },
            measureSvgOverlay: _mcFsMeasureSvgOverlay,
            measureSvgRect:    _mcFsMeasureSvgRect,
            measureHLine:      _mcFsMeasureHLine,
            measureInfoDiv:    _mcFsMeasureInfoDiv,
            measureList:       _mcFsMeasureList,
            measurePreviewMoveHandler: _onMcFsMeasurePreviewMove,
            getSelectedIdx:    function() { return _mcFsSelectedTrendlineIdx; },
            setSelectedIdx:    function(v) { _mcFsSelectedTrendlineIdx = v; },
            trendlines:        _mcFsTrendlines,
            deselectAllTrendlines: _deselectAllTrendlines,
            deselectAllVwaps:  _deselectAllVwaps,
            anchorHitTest:     _anchorHitTest,
            trendlineHitTest:  _trendlineHitTest,
            dragMoveHandler:   _onTrendAnchorDragMove,
            dragEndHandler:    _onTrendAnchorDragEnd,
            getTrendlineMode:  function() { return _mcFsTrendlineMode; },
            setTrendlineMode:  function(v) { _mcFsTrendlineMode = v; },
            getTrendlineStyle: function() { return _mcFsTlMenu.getStyle(); },
            getLastCrosshairTime: function() { return _mcFsLastCrosshairTime; },
            addTrendline:      _addFsTrendline,
            onTrendDrawn:      function(tl) { _cdAddTl(_mcFsSym, tl); },
            onMeasureCommitted: function(ms) { ms.sym = _mcFsBuiltSym || _mcFsSym; _cdAddMs(ms.sym, ms); },
            doneBtnId:         'mc-fs-trendline-btn'
        });
    }

    // ── Trendline SVG preview: mousemove drives the overlay line ──────────
    // Reads raw pixel coords and uses timeToCoordinate/priceToCoordinate to
    // place the anchor — zero LW Charts canvas re-render on every move.
    // Shared trendline-draw-preview / hover-cursor core — used by all three charts
    function _onTrendMouseMoveCore(evt, cfg) {
        // Snap-target dot for the measure tool (see _measureUpdateSnapDot). First, because the branches below return early.
        if (cfg.getMeasureMode && cfg.ohlcv) _measureUpdateSnapDot(evt, cfg);
        // SVG preview during active draw
        if (cfg.getTrendDraw().active) {
            if (!cfg.svgOverlay || !cfg.svgLine || !cfg.candle || !cfg.chart || !cfg.contRef) return;
            var rect  = cfg.contRef.getBoundingClientRect();
            var curX  = evt.clientX - rect.left;
            var curY  = evt.clientY - rect.top;
            var startTime = cfg.getTrendDraw().startTime;
            if (!startTime) return;
            var x1 = cfg.chart.timeScale().timeToCoordinate(startTime);
            var y1 = cfg.candle.priceToCoordinate(cfg.getTrendDraw().startPrice);
            if (x1 == null || y1 == null) return;
            cfg.svgLine.setAttribute('x1', x1);
            cfg.svgLine.setAttribute('y1', y1);
            cfg.svgLine.setAttribute('x2', curX);
            cfg.svgLine.setAttribute('y2', curY);
            return;
        }
        // Cursor feedback when not drawing
        if (cfg.trendlines.length && cfg.contRef && !cfg.getTrendlineMode()) {
            // Don't interfere while an anchor drag is in progress
            if (cfg.getDragState()) return;
            // Grab cursor near anchors of the selected trendline
            var selIdx = cfg.getSelectedIdx();
            if (selIdx !== -1) {
                var anchorSide = cfg.anchorHitTest(evt.clientX, evt.clientY, selIdx);
                if (anchorSide) { cfg.contRef.style.cursor = 'grab'; return; }
            }
            // Pointer cursor on any line body
            var hitIdx = cfg.trendlineHitTest(evt.clientX, evt.clientY);
            cfg.contRef.style.cursor = hitIdx !== -1 ? 'pointer' : '';
        }
    }

    function _onTrendMouseMove(evt) {
        _onTrendMouseMoveCore(evt, {
            getTrendDraw:    function() { return _mcFsTrendDraw; },
            svgOverlay:      _mcFsTrendSvgOverlay,
            svgLine:         _mcFsTrendSvgLine,
            candle:          _mcFsCandle,
            chart:           _mcFsChart,
            contRef:         _mcFsTrendContRef,
            trendlines:      _mcFsTrendlines,
            getTrendlineMode: function() { return _mcFsTrendlineMode; },
            getDragState:    function() { return _mcFsTrendDragState; },
            getSelectedIdx:  function() { return _mcFsSelectedTrendlineIdx; },
            anchorHitTest:   _anchorHitTest,
            trendlineHitTest: _trendlineHitTest,
            ohlcv:           _mcFsOhlcv,
            getMeasureMode:  function() { return _mcFsMeasureMode; },
            getMeasurePhase: function() { return _mcFsMeasurePhase; },
            measureList:     _mcFsMeasureList
        });
    }

    // ── Fullscreen right-click alert context menu ──────────────────────────
    var _mcFsCtxPrice      = null;
    var _mcFsCtxMa         = null; // MA key when right-clicking on an MA line
    var _mcFsCtxTrendline  = null; // {p1, p2} when right-clicking on a trendline
    var _mcFsCtxAvwap      = null; // {anchorIdx, anchorTime} when right-click lands on an AVWAP line
    var _mcFsCtxAttached   = false;

    // Shared "hide the ctx menu" helper — used by all three dismissCtx wrappers
    function _hideCtxMenu(menuId) {
        var el = document.getElementById(menuId);
        if (el) el.style.display = 'none';
    }

    function _mcFsDismissCtx() {
        _hideCtxMenu('mc-fs-ctx-menu');
        _mcFsCtxPrice     = null;
        _mcFsCtxMa        = null;
        _mcFsCtxTrendline = null;
        _mcFsCtxAvwap     = null;
    }

    // Shared "resolve the right-click menu choice into an alert" core — used by
    // all three ctx-alert window functions (alerts.js calls this cross-file).
    function _ctxAlertCore(direction, cfg) {
        var tl = cfg.getCtxTrendline();
        if (tl) {
            cfg.dismiss();
            var sym = cfg.getSym();
            if (!sym) return;
            var tlRes = window.alAddTrendlineAlert(sym, tl.p1, tl.p2, direction);
            // The line now has an alert: show the continuation it is evaluated on.
            if (tlRes && tl.tl) { tl.tl.extend = true; if (tl.tl.requestUpdate) tl.tl.requestUpdate(); }
            return;
        }
        var av = cfg.getCtxAvwap();
        if (av) {
            cfg.dismiss();
            var avSym = cfg.getSym();
            if (!avSym) return;
            window.alAddAvwapAlert(avSym, av.anchorTime, direction);
            return;
        }
        var maKey = cfg.getCtxMa();
        if (maKey) {
            cfg.dismiss();
            var maSym = cfg.getSym();
            if (!maSym) return;
            var openMaForm = function() {
                alShowForm(maSym);
                setTimeout(function() {
                    document.getElementById('al-input-type').value = 'ma';
                    if (typeof alFormTypeChange === 'function') alFormTypeChange();
                    // price_above / price_below are the MA-vs-price condition values
                    document.getElementById('al-input-cond').value = direction === 'above' ? 'price_above' : 'price_below';
                    if (typeof alMACondChange === 'function') alMACondChange();
                    document.getElementById('al-input-ma').value = maKey;
                    document.getElementById('al-input-ma').focus();
                }, 60);
            };
            // MA alerts are evaluated on DAILY bars, but the chart draws each MA on its own timeframe. Clicking the weekly
            // "SMA 50" used to create a bare SMA50 alert that silently watched the daily one (a typical ~8% different line).
            var maTf = cfg.getTf ? cfg.getTf() : 'D';
            if (maTf && maTf !== 'D' && window.alConfirmOpen) {
                var lbl = _maLabel(maKey);
                window.alConfirmOpen('Moving-average alerts use daily bars',
                    'You clicked the ' + (maTf === 'W' ? 'weekly' : 'monthly') + ' ' + lbl + ', but MA alerts always watch the daily average. Set the alert on the daily ' + lbl + ' instead?',
                    openMaForm, 'Use daily ' + lbl, false);
                return;
            }
            openMaForm();
        } else {
            var price = cfg.getCtxPrice();
            cfg.dismiss();
            var pSym = cfg.getSym();
            if (!pSym || price == null) return;
            alShowForm(pSym);
            setTimeout(function() {
                document.getElementById('al-input-type').value = 'price';
                if (typeof alFormTypeChange === 'function') alFormTypeChange();
                document.getElementById('al-input-cond').value = direction;
                document.getElementById('al-input-price').value = price.toFixed(2);
                document.getElementById('al-input-price').focus();
            }, 60);
        }
    }

    window.mcFsCtxAlert = function(direction) {
        _ctxAlertCore(direction, {
            getCtxTrendline: function() { return _mcFsCtxTrendline; },
            getCtxAvwap:     function() { return _mcFsCtxAvwap; },
            getCtxMa:        function() { return _mcFsCtxMa; },
            getCtxPrice:     function() { return _mcFsCtxPrice; },
            getSym:          function() { return _mcFsSym; },
            getTf:           function() { return _mcFsTf; },
            dismiss:         _mcFsDismissCtx
        });
    };

    // ── Watchlist rows inside the chart right-click menu ──────────────────
    // Shared by all three chart menus (fullscreen / watchlist / alerts):
    // _attachCtxMenuCore calls _ctxWlSync() each time it opens one. The rows
    // are built here, once per menu element, so the three copies of the menu
    // markup in index.html don't change. They act on the TICKER, not on the
    // right-clicked price / trendline / AVWAP, so they show for every kind of
    // right-click. The flyout is a DOM child of the menu, so the existing
    // "mousedown outside the menu closes it" check treats clicks inside it as
    // inside. Wrapped in try/catch: a problem here must never stop the alert
    // items from working.
    function _ctxWlEnsure(menu) {
        var host = menu.querySelector('.ctx-wl');
        if (host) return host;
        menu.classList.add('ctx-wl-host');
        host = document.createElement('div');
        host.className = 'ctx-wl';
        host.innerHTML =
            '<div class="ctx-sep"></div>' +
            '<button class="ctx-item ctx-wl-quick" data-act="quick"><span class="ctx-ic"></span><span class="ctx-wl-qtxt"></span></button>' +
            '<div class="ctx-wl-sub">' +
                '<button class="ctx-item" data-act="open"><span class="ctx-ic">\u2261</span>Watchlists<span class="ctx-chev">\u203A</span></button>' +
                '<div class="ctx-fly"><div class="ctx-fly-box"></div></div>' +
            '</div>';
        menu.appendChild(host);

        var sub = host.querySelector('.ctx-wl-sub');
        sub.addEventListener('mouseenter', function() { _ctxWlOpenFly(menu); });
        sub.addEventListener('mouseleave', function() {
            var fly = sub.querySelector('.ctx-fly');
            if (fly.contains(document.activeElement)) return;   // mid-typing a new list name
            fly.style.display = 'none';
        });
        host.addEventListener('click', function(e) {
            var el = e.target.closest('[data-act]');
            if (!el || !host.contains(el)) return;
            var act = el.getAttribute('data-act');
            if (act === 'quick') {
                var last = wlGetLastList();
                if (last) _ctxWlToggle(menu, last);
            } else if (act === 'toggle') {
                _ctxWlToggle(menu, el.getAttribute('data-name'));
            } else if (act === 'open') {
                _ctxWlOpenFly(menu);
            } else if (act === 'new') {
                _ctxWlNewInput(menu);
            }
        });
        return host;
    }

    function _ctxWlRender(menu) {
        var host = menu.querySelector('.ctx-wl');
        var sym  = menu.getAttribute('data-wl-sym');
        if (!host) return;
        if (!sym) { host.style.display = 'none'; return; }
        host.style.display = '';

        var all   = wlGetAll();
        var order = wlGetOrder();
        var last  = wlGetLastList();

        // Quick row: toggles the ticker in the last-used list (same target as the ☆ in the tables)
        var q = host.querySelector('.ctx-wl-quick');
        if (last && all[last]) {
            var has = all[last].indexOf(sym) !== -1;
            var ic  = q.querySelector('.ctx-ic');
            q.style.display = '';
            ic.textContent  = has ? '\u2605' : '\u2606';
            ic.style.color  = has ? 'var(--warning-alt)' : '';
            q.querySelector('.ctx-wl-qtxt').textContent = has ? 'In ' + last : 'Add to ' + last;
        } else {
            q.style.display = 'none';
        }

        // Flyout: every list with a check if the ticker is in it
        var box = host.querySelector('.ctx-fly-box');
        box.innerHTML = '';
        order.forEach(function(name) {
            var on  = all[name] && all[name].indexOf(sym) !== -1;
            var row = document.createElement('button');
            row.className = 'ctx-item' + (on ? ' ctx-on' : '');
            row.setAttribute('data-act', 'toggle');
            row.setAttribute('data-name', name);
            var ck = document.createElement('span');
            ck.className = 'ctx-check';
            ck.textContent = on ? '\u2713' : '';
            var tx = document.createElement('span');
            tx.className = 'ctx-wl-qtxt';
            tx.textContent = name;
            row.appendChild(ck);
            row.appendChild(tx);
            box.appendChild(row);
        });
        if (order.length) {
            var sep = document.createElement('div');
            sep.className = 'ctx-sep';
            box.appendChild(sep);
        }
        var nw = document.createElement('button');
        nw.className = 'ctx-item';
        nw.setAttribute('data-act', 'new');
        nw.innerHTML = '<span class="ctx-check">+</span>New watchlist\u2026';
        box.appendChild(nw);
    }

    function _ctxWlToggle(menu, name) {
        var sym = menu.getAttribute('data-wl-sym');
        if (!sym || !name) return;
        var all = wlGetAll();
        if (!all[name]) all[name] = [];
        var idx = all[name].indexOf(sym);
        if (idx !== -1) {
            all[name].splice(idx, 1);
        } else {
            all[name].push(sym);
            wlSetLastList(name);   // same as the table pickers: the list you add to becomes the quick-add target
        }
        wlSaveAll(all);
        if (currentView === 'watchlists') wlRender();
        wlRefreshStars();
        _ctxWlRender(menu);
    }

    function _ctxWlNewInput(menu) {
        var row = menu.querySelector('.ctx-fly-box [data-act="new"]');
        if (!row) return;
        var input = document.createElement('input');
        input.className   = 'ctx-new-input';
        input.type        = 'text';
        input.maxLength   = 40;
        input.placeholder = 'List name\u2026';
        row.replaceWith(input);
        input.focus();
        input.addEventListener('keydown', function(e) {
            // Keep keys (Delete, letters, arrows) away from the chart's own key handlers.
            // Escape still closes the menu: that listener is capture-phase, so it runs first.
            e.stopPropagation();
            if (e.key !== 'Enter') return;
            var name = input.value.trim();
            var sym  = menu.getAttribute('data-wl-sym');
            if (!name) return;
            var all = wlGetAll();
            if (!all[name]) {
                all[name] = [];
                wlSaveAll(all);
                wlSaveOrder(wlGetOrder());
            }
            if (sym && all[name].indexOf(sym) === -1) {
                all[name].push(sym);
                wlSetLastList(name);
            }
            wlSaveAll(all);
            if (currentView === 'watchlists') wlRender();
            wlRefreshStars();
            _ctxWlRender(menu);
        });
    }

    function _ctxWlOpenFly(menu) {
        var fly = menu.querySelector('.ctx-fly');
        if (!fly) return;
        fly.classList.remove('ctx-flip');
        fly.style.top = '';
        fly.style.display = 'block';
        // Flip to the left of the menu if it would run off the right edge
        if (menu.getBoundingClientRect().right + fly.offsetWidth + 8 > window.innerWidth) fly.classList.add('ctx-flip');
        // Slide up if it would run off the bottom
        var r    = fly.getBoundingClientRect();
        var over = r.bottom - (window.innerHeight - 8);
        if (over > 0) fly.style.top = (-5 - Math.min(over, Math.max(0, r.top - 8))) + 'px';
    }

    // Called by _attachCtxMenuCore BEFORE it shows + measures the menu, so the
    // extra rows are already counted when it clamps the menu to the window.
    function _ctxWlSync(menu, sym) {
        if (!menu || typeof wlGetAll !== 'function' || typeof wlGetOrder !== 'function' || typeof wlSaveAll !== 'function') return;
        try {
            _ctxWlEnsure(menu);
            menu.setAttribute('data-wl-sym', sym || '');
            var fly = menu.querySelector('.ctx-fly');
            if (fly) fly.style.display = 'none';
            _ctxWlRender(menu);
        } catch (e) {}
    }

    // ── Shared right-click context-menu core — used by all three charts.
    // IMPORTANT: the attach wrapper only runs its setup once (guarded by
    // getAttached/setAttached); the actual 'contextmenu' listener it registers
    // here persists across chart rebuilds. So every piece of state that gets
    // *reassigned* on a chart rebuild (chart, candle, trendlines, vwapSeries,
    // ohlcv, maDataMap, maSeries, svgOverlay, sym, ...) MUST be read through a
    // getter function each time the listener fires — capturing the value
    // directly here would freeze it at first-attach time and go stale the
    // moment the user switches symbols. (alerts.js calls this cross-file.)
    function _attachCtxMenuCore(cfg) {
        if (cfg.getAttached()) return;
        cfg.setAttached(true);
        var parentEl = document.getElementById(cfg.parentElId);
        var chartDiv = document.getElementById(cfg.chartDivId);
        parentEl.addEventListener('contextmenu', function(evt) {
            if (!chartDiv.contains(evt.target)) return; // header / settings bar click
            evt.preventDefault();
            evt.stopPropagation();
            // Toggle off data tooltip on right-click
            if (cfg.getTooltipEnabled()) {
                cfg.setTooltipEnabled(false);
                var _ttBtn = document.getElementById(cfg.tooltipBtnId);
                if (_ttBtn) _ttBtn.classList.remove('active');
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
            }
            // Right-click: cancel active measurement first (no context menu shown)
            if (cfg.getMeasurePhase() === 1) {
                cfg.setMeasureActive(false);
                cfg.setMeasurePhase(0);
                if (cfg.getMeasureRafId()) { cancelAnimationFrame(cfg.getMeasureRafId()); cfg.setMeasureRafId(null); }
                document.removeEventListener('mousemove', cfg.measurePreviewMoveHandler);
                _hideMeasureOverlay(cfg.getMeasureSvgOverlay(), cfg.getMeasureInfoDiv());
                cfg.setMeasureResult(null);
                return;
            }
            if (cfg.getMeasureResult()) {
                _hideMeasureOverlay(cfg.getMeasureSvgOverlay(), cfg.getMeasureInfoDiv());
                cfg.setMeasureResult(null);
                return;
            }
            // Right-click while drawing a trendline: cancel draw, skip context menu
            var trendDraw = cfg.trendDraw;
            if (trendDraw.active) {
                trendDraw.active = false; trendDraw.startTime = null; trendDraw.startPrice = null;
                var tdSvg = cfg.getSvgOverlay();
                if (tdSvg) tdSvg.style.display = 'none';
                return;
            }
            // Right-click while AVWAP mode is active: turn it off, skip context menu
            if (cfg.getVwapMode()) {
                cfg.setVwapMode(false);
                var vBtn = document.getElementById(cfg.vwapBtnId);
                if (vBtn) vBtn.classList.remove('active');
                return;
            }
            if (!cfg.getChart() || !cfg.getSym()) return;
            // ── Trendline right-click: check hit before price/MA ──────────────
            var _tlHitIdx = cfg.trendlineHitTest(evt.clientX, evt.clientY);
            if (_tlHitIdx !== -1) {
                var _tlHit = cfg.getTrendlines()[_tlHitIdx];
                cfg.setCtxTrendline({ p1: _tlHit.leftP, p2: _tlHit.rightP, tl: _tlHit });
                cfg.setCtxPrice(null);
                cfg.setCtxMa(null);
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Alert above trendline';
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Alert below trendline';
                var _tlMenu = document.getElementById(cfg.ctxMenuId);
                _ctxWlSync(_tlMenu, cfg.getSym());
                _tlMenu.style.display = 'block';
                var mw = _tlMenu.offsetWidth  || 185;
                var mh = _tlMenu.offsetHeight || 90;
                var x  = Math.min(evt.clientX, window.innerWidth  - mw - 8);
                var y  = Math.min(evt.clientY, window.innerHeight - mh - 8);
                _tlMenu.style.left = x + 'px';
                _tlMenu.style.top  = y + 'px';
                setTimeout(function() {
                    function _tlDismiss(e) {
                        if (!_tlMenu.contains(e.target)) {
                            cfg.dismissCtx();
                            document.removeEventListener('mousedown', _tlDismiss, true);
                            document.removeEventListener('keydown',   _tlKd,      true);
                        }
                    }
                    function _tlKd(e) {
                        if (e.key === 'Escape') {
                            cfg.dismissCtx();
                            document.removeEventListener('mousedown', _tlDismiss, true);
                            document.removeEventListener('keydown',   _tlKd,      true);
                        }
                    }
                    document.addEventListener('mousedown', _tlDismiss, true);
                    document.addEventListener('keydown',   _tlKd,      true);
                }, 0);
                return;
            }
            // ── AVWAP right-click: check hit before price/MA ──────────────────
            var _avHitIdx = cfg.vwapHitTest(evt.clientX, evt.clientY);
            if (_avHitIdx !== -1) {
                var vwapSeries = cfg.getVwapSeries();
                var ohlcv      = cfg.getOhlcv();
                var _avHit = vwapSeries[_avHitIdx];
                cfg.setCtxAvwap({ anchorIdx: _avHit.anchor, anchorTime: ohlcv[_avHit.anchor] ? ohlcv[_avHit.anchor].time : null });
                cfg.setCtxTrendline(null);
                cfg.setCtxPrice(null);
                cfg.setCtxMa(null);
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Alert above AVWAP';
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Alert below AVWAP';
                var _avMenu = document.getElementById(cfg.ctxMenuId);
                _ctxWlSync(_avMenu, cfg.getSym());
                _avMenu.style.display = 'block';
                var avMw = _avMenu.offsetWidth  || 185;
                var avMh = _avMenu.offsetHeight || 90;
                var avX  = Math.min(evt.clientX, window.innerWidth  - avMw - 8);
                var avY  = Math.min(evt.clientY, window.innerHeight - avMh - 8);
                _avMenu.style.left = avX + 'px';
                _avMenu.style.top  = avY + 'px';
                setTimeout(function() {
                    function _avDismiss(e) {
                        if (!_avMenu.contains(e.target)) {
                            cfg.dismissCtx();
                            document.removeEventListener('mousedown', _avDismiss, true);
                            document.removeEventListener('keydown',   _avKd,      true);
                        }
                    }
                    function _avKd(e) {
                        if (e.key === 'Escape') {
                            cfg.dismissCtx();
                            document.removeEventListener('mousedown', _avDismiss, true);
                            document.removeEventListener('keydown',   _avKd,      true);
                        }
                    }
                    document.addEventListener('mousedown', _avDismiss, true);
                    document.addEventListener('keydown',   _avKd,      true);
                }, 0);
                return;
            }
            var chartRect = chartDiv.getBoundingClientRect();
            var localY    = evt.clientY - chartRect.top;
            var price = cfg.getLastCrosshairPrice();
            if (price == null || isNaN(price)) {
                // Fallback: crosshair is in empty space to the right of the last candle —
                // LW Charts never fires crosshair data there, so derive the price
                // directly from the click's Y coordinate via the candle series.
                var candle = cfg.getCandle();
                if (candle) {
                    var fallbackPrice = candle.coordinateToPrice(localY);
                    if (fallbackPrice != null && !isNaN(fallbackPrice)) price = fallbackPrice;
                }
            }
            if (price == null || isNaN(price)) return;

            // MA proximity — find the closest active MA within the hit threshold
            var nearestMa   = null;
            var nearestDist = 10; // px
            var lastCrosshairTime = cfg.getLastCrosshairTime();
            if (lastCrosshairTime) {
                var maDataMap = cfg.getMaDataMap();
                var maSeries  = cfg.getMaSeries();
                Object.keys(maDataMap).forEach(function(key) {
                    if (!maSeries[key]) return;
                    var maVal = maDataMap[key].get(lastCrosshairTime);
                    if (maVal == null) return;
                    var maCoord = maSeries[key].priceToCoordinate(maVal);
                    if (maCoord == null) return;
                    var dist = Math.abs(localY - maCoord);
                    if (dist < nearestDist) { nearestDist = dist; nearestMa = key; }
                });
            }

            cfg.setCtxPrice(price);
            cfg.setCtxMa(nearestMa);

            if (nearestMa) {
                var maLabel = _maLabel(nearestMa);
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Price crosses above ' + maLabel;
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Price crosses below ' + maLabel;
            } else {
                var fmt = '$' + price.toFixed(2);
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Alert above ' + fmt;
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Alert below ' + fmt;
            }
            var menu  = document.getElementById(cfg.ctxMenuId);
            _ctxWlSync(menu, cfg.getSym());
            menu.style.display = 'block';
            var mw = menu.offsetWidth  || 185;
            var mh = menu.offsetHeight || 90;
            var x  = Math.min(evt.clientX, window.innerWidth  - mw - 8);
            var y  = Math.min(evt.clientY, window.innerHeight - mh - 8);
            menu.style.left = x + 'px';
            menu.style.top  = y + 'px';
            setTimeout(function() {
                function _dismiss(e) {
                    if (!menu.contains(e.target)) {
                        cfg.dismissCtx();
                        document.removeEventListener('mousedown', _dismiss, true);
                        document.removeEventListener('keydown',   _kd,      true);
                    }
                }
                function _kd(e) {
                    if (e.key === 'Escape') {
                        cfg.dismissCtx();
                        document.removeEventListener('mousedown', _dismiss, true);
                        document.removeEventListener('keydown',   _kd,      true);
                    }
                }
                document.addEventListener('mousedown', _dismiss, true);
                document.addEventListener('keydown',   _kd,      true);
            }, 0);
        }, true); // capture phase — overlay-level intercept, nothing below can block it
    }

    function _mcFsAttachCtxMenu() {
        _attachCtxMenuCore({
            getAttached: function() { return _mcFsCtxAttached; },
            setAttached: function(v) { _mcFsCtxAttached = v; },
            parentElId: 'mc-fullscreen-overlay',
            chartDivId: 'mc-fullscreen-chart',
            getTooltipEnabled: function() { return _mcFsTooltipEnabled; },
            setTooltipEnabled: function(v) { _mcFsTooltipEnabled = v; },
            tooltipBtnId: 'mc-fs-tooltip-btn',
            getMeasurePhase:  function() { return _mcFsMeasurePhase; },
            setMeasurePhase:  function(v) { _mcFsMeasurePhase = v; },
            setMeasureActive: function(v) { _mcFsMeasureActive = v; },
            getMeasureRafId:  function() { return _mcFsMeasureRafId; },
            setMeasureRafId:  function(v) { _mcFsMeasureRafId = v; },
            measurePreviewMoveHandler: _onMcFsMeasurePreviewMove,
            getMeasureSvgOverlay: function() { return _mcFsMeasureSvgOverlay; },
            getMeasureInfoDiv:    function() { return _mcFsMeasureInfoDiv; },
            getMeasureResult: function() { return _mcFsMeasureResult; },
            setMeasureResult: function(v) { _mcFsMeasureResult = v; },
            trendDraw:     _mcFsTrendDraw,
            getSvgOverlay: function() { return _mcFsTrendSvgOverlay; },
            getVwapMode: function() { return _mcFsVwapMode; },
            setVwapMode: function(v) { _mcFsVwapMode = v; },
            vwapBtnId: 'mc-fs-vwap-btn',
            getChart: function() { return _mcFsChart; },
            getSym:   function() { return _mcFsSym; },
            trendlineHitTest: _trendlineHitTest,
            getTrendlines: function() { return _mcFsTrendlines; },
            ctxAboveTxtId:  'mc-fs-ctx-above-txt',
            ctxBelowTxtId:  'mc-fs-ctx-below-txt',
            ctxMenuId:      'mc-fs-ctx-menu',
            setCtxTrendline: function(v) { _mcFsCtxTrendline = v; },
            setCtxPrice:     function(v) { _mcFsCtxPrice = v; },
            setCtxMa:        function(v) { _mcFsCtxMa = v; },
            vwapHitTest: _mcFsVwapHitTest,
            getVwapSeries: function() { return _mcFsVwapSeries; },
            getOhlcv:      function() { return _mcFsOhlcv; },
            setCtxAvwap: function(v) { _mcFsCtxAvwap = v; },
            getCandle: function() { return _mcFsCandle; },
            getLastCrosshairPrice: function() { return _mcFsLastCrosshairPrice; },
            getLastCrosshairTime:  function() { return _mcFsLastCrosshairTime; },
            getMaDataMap: function() { return _mcFsMaDataMap; },
            getMaSeries:  function() { return _mcFsMaSeries; },
            dismissCtx: _mcFsDismissCtx
        });
    }

    // ── Live quote helper ─────────────────────────────────────────────────────
    // quotes_batch answers in two shapes: Questrade's {ticker, price, dayHigh, dayLow}, and — when
    // Questrade has nothing live — Yahoo's {symbol, price: null, regularMarketPrice, ...}. The
    // watchlist / alerts / scans pollers already accept both; the fullscreen paths below used to read
    // only q.price and silently ignored the Yahoo shape. Returns { price, dayHigh, dayLow } or null.
    // (regularMarketDayHigh/Low are Yahoo's standard field names; they are optional here — absent
    // just means null, same as before.)
    function _mcQuoteFromBatch(data) {
        var q = data && data.quotes && data.quotes[0];
        if (!q) return null;
        var price = q.price != null ? q.price : q.regularMarketPrice;
        if (!price) return null;
        var dh = q.dayHigh != null ? q.dayHigh : (q.regularMarketDayHigh != null ? q.regularMarketDayHigh : null);
        var dl = q.dayLow  != null ? q.dayLow  : (q.regularMarketDayLow  != null ? q.regularMarketDayLow  : null);
        return { price: price, dayHigh: dh || null, dayLow: dl || null };
    }

    // ── Live bar injection — fullscreen + WL charts ───────────────────────────
    // Folds today's price into the chart's last bar (Daily: today's bar; Weekly/Monthly: the current
    // period's bar, see _mcApplyLiveWM).
    //
    // Market OPEN: asks the proxy for ONE fresh quote for this symbol first (the same quotes_batch call
    // the fullscreen/alerts live ticks make) rather than trusting the in-memory caches. Those caches
    // (indLivePrices / wlLivePrices / scanLivePrices / alertPrices / snapshot row.price) carry no
    // freshness check, indLivePrices is never cleared when you leave the Industry-stocks view, and the
    // others only refresh while their own view is active — so opening fullscreen from Watchlist or
    // Alerts could seed the chart with a price minutes old. If the fresh quote can't be had (request
    // failed, nothing usable in the reply) it falls back to the ORIGINAL resolution below (caches, then
    // a 2-day proxy fetch), so it is never worse than before.
    // Market CLOSED: unchanged — original resolution only.
    function _injectChartLiveBar(sym, tf, candle, vol, ohlcvArr, isStale) {
        if (!candle || !ohlcvArr || !ohlcvArr.length) return;

        function _applyLiveBar(p, dh, dl) {
            if (!p || !candle || !ohlcvArr.length) return;
            if (tf !== 'D') {
                var _wm = _mcApplyLiveWM(ohlcvArr, tf, p, dh, dl, true);
                if (!_wm) return;
                try { candle.update(_wm); } catch(e) {}
                if (vol) {
                    try { vol.update({ time: _wm.time, value: _wm.volume, color: p >= _wm.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') }); } catch(e) {}
                }
                return;
            }
            var now = new Date();
            // Use noon UTC (midnight UTC + 43200s) to match the noon-UTC stamps from fetchMcOhlcv.
            var todayTs = Math.floor(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 1000) + 43200;
            var last = ohlcvArr[ohlcvArr.length - 1];
            var lastDayTs = Math.floor(last.time / 86400) * 86400 + 43200;
            var open, high, low, volume;
            if (lastDayTs === todayTs) {
                open   = last.open;
                high   = dh != null ? Math.max(last.high, dh, p) : Math.max(last.high, p);
                low    = dl != null ? Math.min(last.low,  dl, p) : Math.min(last.low,  p);
                volume = last.volume;
                last.high  = high;
                last.low   = low;
                last.close = p;
            } else {
                if (!wlIsMarketOpen()) return;
                open = high = low = p; volume = 0;
                ohlcvArr.push({ time: todayTs, open: open, high: high, low: low, close: p, volume: volume });
            }
            try { candle.update({ time: todayTs, open: open, high: high, low: low, close: p, volume: volume }); } catch(e) {}
            if (vol) {
                try { vol.update({ time: todayTs, value: volume, color: p >= open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') }); } catch(e) {}
            }
        }

        // Original resolution (unchanged): in-memory live caches, then a 2-day proxy fetch.
        function _applyFromCaches() {
            var price = null, dayHigh = null, dayLow = null;

            // 1. indLivePrices — richest: has dayHigh + dayLow
            if (typeof indLivePrices !== 'undefined' && indLivePrices[sym]) {
                var _lp = indLivePrices[sym];
                price   = _lp.price   || null;
                dayHigh = _lp.dayHigh || null;
                dayLow  = _lp.dayLow  || null;
            }
            // 2. wlLivePrices — price only
            if (!price && wlLivePrices && wlLivePrices[sym]) {
                price = wlLivePrices[sym].price || null;
            }
            // 3. snapshot row — price only
            if (!price && snapshot && snapshot.by_industry) {
                outerILB: for (var _ii in snapshot.by_industry) {
                    var _rr = snapshot.by_industry[_ii];
                    for (var _jj = 0; _jj < _rr.length; _jj++) {
                        if (_rr[_jj].ticker === sym) { price = _rr[_jj].price || null; break outerILB; }
                    }
                }
            }

            if (price) {
                _applyLiveBar(price, dayHigh, dayLow);
            } else {
                // Fallback: fresh 2-day proxy quote — guards against a stale close
                fetch(WL_PROXY + '?symbol=' + encodeURIComponent(sym) + '&interval=1d&range=2d')
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (isStale && isStale()) return; // chart was replaced
                        var result = data && data.chart && data.chart.result && data.chart.result[0];
                        if (!result) return;
                        var meta = result.meta || {};
                        var lp   = meta.regularMarketPrice;
                        if (lp) _applyLiveBar(lp, meta.regularMarketDayHigh || null, meta.regularMarketDayLow || null);
                    }).catch(function() {});
            }
        }

        if (!wlIsMarketOpen()) { _applyFromCaches(); return; }

        fetch(WL_PROXY + '?action=quotes_batch&tickers=' + encodeURIComponent(sym))
            .then(function(r) { return r.ok ? r.json() : null; })
            .catch(function() { return null; })   // network/parse failure -> null -> original fallback below
            .then(function(data) {
                if (isStale && isStale()) return; // chart was replaced while the quote was in flight
                var fq = _mcQuoteFromBatch(data);
                if (fq) _applyLiveBar(fq.price, fq.dayHigh, fq.dayLow);
                else    _applyFromCaches();
            });
    }

    // ── Fullscreen live tick ─────────────────────────────────────────────────
    // _injectChartLiveBar above fires once, on open, sourcing from indLivePrices
    // / wlLivePrices — but every one of those caches' own pollers explicitly
    // stands down while _mcFsIsOpen() is true, so they go stale for as long as
    // fullscreen stays open. Re-reading them on a timer would just keep
    // reapplying the same one price. This does a direct, single-ticker fetch
    // each tick instead — cheap (one ticker, not a 30-ticker batch) since it's
    // replacing the paused pollers' budget, not competing with it.
    function _mcFsStartLiveTick(sym, tf) {
        _mcFsStopLiveTick();
        if (!wlIsMarketOpen()) return;
        _mcFsLiveTimer = setInterval(function() {
            if (!_mcFsIsOpen() || _mcFsSym !== sym || !_mcFsCandle) { _mcFsStopLiveTick(); return; }
            if (!wlIsMarketOpen()) { _mcFsStopLiveTick(); return; }
            fetch(WL_PROXY + '?action=quotes_batch&tickers=' + encodeURIComponent(sym))
                .then(function(r) { return r.ok ? r.json() : null; })
                .then(function(data) {
                    var q = _mcQuoteFromBatch(data); // accepts both the Questrade and the Yahoo-fallback quote shape
                    if (!q || _mcFsSym !== sym || !_mcFsCandle || !_mcFsOhlcv.length) return;
                    if (tf !== 'D') {
                        // W/M: fold into the current period's bar; never create one mid-tick
                        var _wm = _mcApplyLiveWM(_mcFsOhlcv, tf, q.price, q.dayHigh, q.dayLow, false);
                        if (_wm) { try { _mcFsCandle.update(_wm); } catch(e) {} }
                        return;
                    }
                    var now = new Date();
                    var todayTs = Math.floor(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 1000) + 43200;
                    var last = _mcFsOhlcv[_mcFsOhlcv.length - 1];
                    var lastDayTs = Math.floor(last.time / 86400) * 86400 + 43200;
                    if (lastDayTs !== todayTs) return; // new trading day — let the next open/reload pick it up
                    var high = q.dayHigh != null ? Math.max(last.high, q.dayHigh, q.price) : Math.max(last.high, q.price);
                    var low  = q.dayLow  != null ? Math.min(last.low,  q.dayLow,  q.price) : Math.min(last.low,  q.price);
                    last.high = high; last.low = low; last.close = q.price;
                    try { _mcFsCandle.update({ time: todayTs, open: last.open, high: high, low: low, close: q.price, volume: last.volume }); } catch(e) {}
                }).catch(function() {});
        }, 10 * 1000);
    }

    function _mcFsStopLiveTick() {
        if (_mcFsLiveTimer) { clearInterval(_mcFsLiveTimer); _mcFsLiveTimer = null; }
    }

    // ── Pre/post-market badge (fullscreen chart) ────────────────────────────
    // Yahoo's chart-endpoint meta (what _mcMetaCache holds) never includes
    // marketState/extended-hours prices — those only exist on Yahoo's quote
    // endpoint, which now requires crumb/cookie auth. Instead the Worker
    // derives state + price from the same chart endpoint via
    // includePrePost=true + currentTradingPeriod (see Worker action=prepost).
    // One fetch per fullscreen open/TF-switch — not part of the OHLCV queue.
    function fetchMcPrePost(sym) {
        return fetch(WL_PROXY + '?action=prepost&symbol=' + encodeURIComponent(sym))
            .then(function(r) { return r.json(); })
            .catch(function() { return null; });
    }

    // Shared pre/post-market badge renderer — used by the fullscreen and
    // watchlist charts here, and called cross-file by the alerts chart too
    // (same pattern as fetchMcOhlcv/fetchMcPrePost already being shared).
    var _prePostTokens = {}; // per-badge-id guard counters, keyed by badgeId

    function _renderPrePostBadge(cfg) {
        // cfg: { sym, badgeId, getChart(), getCurrentSym(), isOpen() }
        var badge = document.getElementById(cfg.badgeId);
        if (badge) badge.style.display = 'none'; // hidden until fetch confirms PRE/POST
        var token = (_prePostTokens[cfg.badgeId] = (_prePostTokens[cfg.badgeId] || 0) + 1);
        fetchMcPrePost(cfg.sym).then(function(data) {
            if (token !== _prePostTokens[cfg.badgeId]) return; // superseded by a later open/switch
            if (cfg.getCurrentSym() !== cfg.sym) return;         // symbol changed under us
            if (!cfg.isOpen()) return;
            var badge = document.getElementById(cfg.badgeId);
            if (!badge || !data) return;

            var state  = data.marketState;
            var isPre  = state === 'PRE';
            var isPost = state === 'POST';
            if ((!isPre && !isPost) || typeof data.price !== 'number') { badge.style.display = 'none'; return; }

            function fp(v) { return v != null ? v.toFixed(2) : '—'; }
            var price = data.price, chg = data.change, pct = data.changePercent;
            var up    = chg == null || chg >= 0;
            var sign  = up ? '+' : '';
            var label = isPre ? 'Pre-Mkt' : 'Post-Mkt';

            badge.style.color = up ? 'var(--success)' : 'var(--danger)';
            badge.innerHTML =
                '<span style="color:var(--text-muted-2);font-weight:500;">' + label + '</span>&nbsp; ' +
                fp(price) +
                (chg != null ? '&nbsp;' + sign + fp(chg) : '') +
                (pct != null ? '&nbsp;(' + sign + pct.toFixed(2) + '%)' : '');

            // Offset from the actual rendered price-scale width so the badge
            // never collides with axis labels, regardless of price range/digits.
            var rightOffset = 8;
            var chart = cfg.getChart();
            if (chart) {
                try { rightOffset = chart.priceScale('right').width() + 2; } catch (e) {}
            }
            badge.style.right = rightOffset + 'px';
            badge.style.display = 'block';
        });
    }

    function _mcFsRenderPrePostBadge(sym) {
        _renderPrePostBadge({
            sym: sym,
            badgeId: 'mc-fs-prepost-badge',
            getChart: function() { return _mcFsChart; },
            getCurrentSym: function() { return _mcFsSym; },
            isOpen: function() {
                var overlay = document.getElementById('mc-fullscreen-overlay');
                return !!overlay && overlay.classList.contains('open');
            }
        });
    }

    function _wlRenderPrePostBadge(sym) {
        _renderPrePostBadge({
            sym: sym,
            badgeId: 'wl-chart-prepost-badge',
            getChart: function() { return _wlChart; },
            getCurrentSym: function() { return _wlSym; },
            isOpen: function() {
                var wlContainer = document.getElementById('wl-chart-widget');
                return !!wlContainer && wlContainer.style.display !== 'none';
            }
        });
    }

    // ── Fullscreen screenshot (Alt+S) ─────────────────────────────────────────
    // Output, top to bottom: (1) the fullscreen header row (symbol, industry, RS/EPS
    // badges, fundamentals); (2) the market info bar from the right side of the settings
    // row (price/change, ADR, mkt cap, day range, 52W range); (3) the chart exactly as
    // drawn (candles, volume, MAs, AVWAPs, trendlines, watermark, axes) plus any committed measurements
    // (an SVG overlay, redrawn onto the image by _mcFsShotPaintMeasures). Not included:
    // the timeframe/MA/trendline/measure/AVWAP buttons, the Details / + / x buttons, the
    // hover OHLC legend, and the crosshair.
    // The two top rows are drawn _MC_SHOT_STRIP_ZOOM times larger than on screen but keep their combined
    // on-screen height (image is never taller than before); the fundamentals are pushed to the right
    // edge to mirror the symbol on the left.
    // Goes to the clipboard as a PNG. Always captured at the screen's native pixel
    // density, never upscaled; refused if the result would be under the width below.
    var _MC_SHOT_MIN_WIDTH = 1600;   // real pixels
    // The header row and market info bar are drawn this many times larger than on screen so they
    // stay readable when the image is shrunk by a chat app / post. The chart itself is not resized.
    // The two rows keep EXACTLY their on-screen combined height (so the image is never taller than
    // before); the larger content is centred in each row and the row padding absorbs the difference.
    // The zoom is automatically reduced (never below 1) if the padding would drop below
    // _MC_SHOT_STRIP_MIN_PAD, or if the larger rows wouldn't fit the image width.
    var _MC_SHOT_STRIP_ZOOM    = 1.15;
    var _MC_SHOT_STRIP_MIN_PAD = 3;    // CSS px of breathing room kept above / below each row's content
    var _mcFsShotBusy      = false;

    function _mcFsShotToast(msg, ok) {
        var el = document.getElementById('mc-fs-shot-toast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'mc-fs-shot-toast';
            el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:100000;' +
                'padding:8px 16px;border-radius:6px;font-size:12px;font-weight:600;pointer-events:none;' +
                'background:var(--bg-subtle);border:1px solid var(--bg-surface);box-shadow:0 6px 20px rgba(0,0,0,0.35);';
            document.body.appendChild(el);
        }
        el.textContent = msg;
        el.style.color = ok ? 'var(--success)' : 'var(--danger)';
        el.style.display = 'block';
        clearTimeout(el._t);
        el._t = setTimeout(function() { el.style.display = 'none'; }, ok ? 2200 : 4200);
    }

    // Paints the live header DOM onto a canvas (no library needed): backgrounds,
    // borders and text are read from computed styles, and every character is placed
    // at the x/y the browser itself laid it out at, so the result matches the screen.
    // ctx must already carry a transform that maps viewport CSS px -> canvas px.
    // root = element to paint; skip = elements to leave out; only = if given, paint root's
    // own background/border and then just this one descendant (used for the market info bar).
    function _mcFsShotPaint(ctx, root, skip, only) {
        function px(v) { return parseFloat(v) || 0; }
        function clear(c) { return !c || c === 'transparent' || /^rgba\(\s*\d+,\s*\d+,\s*\d+,\s*0\s*\)$/.test(c); }
        function rrect(x, y, w, h, r) {
            ctx.beginPath();
            if (r > 0 && ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
        }

        function drawBox(cs, r, alpha) {
            ctx.globalAlpha = alpha;
            var rad = px(cs.borderTopLeftRadius);
            var sh = /^(rgba?\([^)]*\))\s+(-?[\d.]+)px\s+(-?[\d.]+)px\s+([\d.]+)px(?:\s+([\d.]+)px)?/.exec(cs.boxShadow || '');
            if (sh && !clear(sh[1]) && px(sh[4]) === 0 && px(sh[5]) > 0) {
                var sp = px(sh[5]);
                ctx.fillStyle = sh[1];
                rrect(r.left + px(sh[2]) - sp, r.top + px(sh[3]) - sp, r.width + sp * 2, r.height + sp * 2, rad + sp);
                ctx.fill();
            }
            if (!clear(cs.backgroundColor)) {
                ctx.fillStyle = cs.backgroundColor;
                rrect(r.left, r.top, r.width, r.height, rad);
                ctx.fill();
            }
            var bt = px(cs.borderTopWidth), br = px(cs.borderRightWidth), bb = px(cs.borderBottomWidth), bl = px(cs.borderLeftWidth);
            if (!(bt || br || bb || bl)) return;
            if (bt && bt === br && br === bb && bb === bl && cs.borderTopStyle !== 'none' && !clear(cs.borderTopColor)) {
                ctx.strokeStyle = cs.borderTopColor;
                ctx.lineWidth = bt;
                rrect(r.left + bt / 2, r.top + bt / 2, r.width - bt, r.height - bt, Math.max(rad - bt / 2, 0));
                ctx.stroke();
                return;
            }
            if (bt && cs.borderTopStyle    !== 'none' && !clear(cs.borderTopColor))    { ctx.fillStyle = cs.borderTopColor;    ctx.fillRect(r.left, r.top, r.width, bt); }
            if (bb && cs.borderBottomStyle !== 'none' && !clear(cs.borderBottomColor)) { ctx.fillStyle = cs.borderBottomColor; ctx.fillRect(r.left, r.bottom - bb, r.width, bb); }
            if (bl && cs.borderLeftStyle   !== 'none' && !clear(cs.borderLeftColor))   { ctx.fillStyle = cs.borderLeftColor;   ctx.fillRect(r.left, r.top, bl, r.height); }
            if (br && cs.borderRightStyle  !== 'none' && !clear(cs.borderRightColor))  { ctx.fillStyle = cs.borderRightColor;  ctx.fillRect(r.right - br, r.top, br, r.height); }
        }

        function drawText(tn, alpha) {
            var txt = tn.nodeValue;
            if (!txt || !txt.trim()) return;
            var par = tn.parentElement;
            if (!par) return;
            var cs = getComputedStyle(par);
            if (cs.visibility === 'hidden') return;
            ctx.globalAlpha = alpha;
            ctx.font = cs.fontStyle + ' ' + cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
            ctx.fillStyle = cs.color;
            ctx.textBaseline = 'alphabetic';
            var m = ctx.measureText('Hg');
            var asc = m.fontBoundingBoxAscent, desc = m.fontBoundingBoxDescent;
            var tt = cs.textTransform;
            var range = document.createRange();
            for (var i = 0; i < txt.length; i++) {
                var ch = txt.charAt(i);
                if (/\s/.test(ch)) continue;
                range.setStart(tn, i);
                range.setEnd(tn, i + 1);
                var cr = range.getBoundingClientRect();
                if (!cr.width && !cr.height) continue;
                if (tt === 'uppercase') ch = ch.toUpperCase(); else if (tt === 'lowercase') ch = ch.toLowerCase();
                ctx.fillText(ch, cr.left, cr.top + (cr.height - (asc + desc)) / 2 + asc);
            }
        }

        function walk(node, alpha) {
            if (node.nodeType === 3) { drawText(node, alpha); return; }
            if (node.nodeType !== 1 || skip.indexOf(node) !== -1) return;
            var tag = node.nodeName.toLowerCase();
            if (tag === 'svg' || tag === 'img' || tag === 'input' || tag === 'button') return;
            var cs = getComputedStyle(node);
            if (cs.display === 'none' || cs.visibility === 'hidden') return;
            var r = node.getBoundingClientRect();
            if (!r.width && !r.height) return;
            var op = parseFloat(cs.opacity);
            var a  = alpha * (isNaN(op) ? 1 : op);
            drawBox(cs, r, a);
            var clip = cs.overflowX !== 'visible' || cs.overflowY !== 'visible';
            if (clip) { ctx.save(); ctx.beginPath(); ctx.rect(r.left, r.top, r.width, r.height); ctx.clip(); }
            for (var c = node.firstChild; c; c = c.nextSibling) walk(c, a);
            if (clip) ctx.restore();
        }

        if (only) {
            drawBox(getComputedStyle(root), root.getBoundingClientRect(), 1);
            walk(only, 1);
        } else {
            walk(root, 1);
        }
        ctx.globalAlpha = 1;
    }

    // Right edge (viewport CSS px) of the last visible character inside el, or null if there is none.
    // Measures the glyphs themselves, so trailing padding / margins in el can't skew the result.
    function _mcFsShotInkRight(el) {
        var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        var right = null, tn;
        while ((tn = walker.nextNode())) {
            var t = tn.nodeValue;
            if (!t || !t.trim()) continue;
            var i = t.length - 1;
            while (i > 0 && /\s/.test(t.charAt(i))) i--;
            var rg = document.createRange();
            rg.setStart(tn, i);
            rg.setEnd(tn, i + 1);
            var r = rg.getBoundingClientRect();
            if (r.width && (right === null || r.right > right)) right = r.right;
        }
        return right;
    }

    // Vertical extent (viewport CSS px) of everything inside the given elements, including parts that
    // overflow their parent (e.g. the small % labels above the range bars). Hidden elements are ignored.
    // Returns {top, bottom} or null if nothing is visible.
    function _mcFsShotExtentY(els) {
        var top = Infinity, bottom = -Infinity;
        els.forEach(function(el) {
            if (!el || getComputedStyle(el).display === 'none') return;
            var all = [el].concat(Array.prototype.slice.call(el.querySelectorAll('*')));
            all.forEach(function(n) {
                var r = n.getBoundingClientRect();
                if (!r.width && !r.height) return;
                if (r.top < top) top = r.top;
                if (r.bottom > bottom) bottom = r.bottom;
            });
        });
        return bottom > top ? { top: top, bottom: bottom } : null;
    }

    // The committed measurements are an SVG overlay, which the chart library's takeScreenshot() cannot see (it
    // only captures its own canvases), so they are redrawn here from the live SVG elements: same geometry, same
    // resolved colours, same labels. Always drawn at the normal (unselected) thickness so a selected measurement
    // doesn't come out bold in the image. Items scrolled off the visible scale (box === null) are skipped.
    // The in-progress drag preview is not included. scale = canvas px per CSS px; offY = canvas y where the
    // chart image starts.
    function _mcFsShotPaintMeasures(ctx, list, scale, offY) {
        if (!list || !list.length) return;
        function cs(el, prop) {
            var v = getComputedStyle(el)[prop];
            return (!v || v === 'none') ? null : v;
        }
        function n(el, a) { return parseFloat(el.getAttribute(a)) || 0; }
        function seg(el, width, dash) {
            var c = cs(el, 'stroke');
            if (!c) return;
            var op = parseFloat(getComputedStyle(el).strokeOpacity);   // the dashed arrow is drawn fainter than the lines
            ctx.globalAlpha = isFinite(op) ? op : 1;
            ctx.beginPath();
            ctx.moveTo(n(el, 'x1'), n(el, 'y1'));
            ctx.lineTo(n(el, 'x2'), n(el, 'y2'));
            ctx.strokeStyle = c;
            ctx.lineWidth   = width;
            ctx.setLineDash(dash || []);
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
        ctx.save();
        ctx.setTransform(scale, 0, 0, scale, 0, offY);
        list.forEach(function(m) {
            if (!m || !m.box || !m.topLine || !m.topLine.isConnected) return;
            try {
                seg(m.topLine, 1.5);
                seg(m.botLine, 1.5);
                seg(m.vLine, 1, [3, 3]);
                var hc  = cs(m.head, 'fill');
                var pts = (m.head.getAttribute('points') || '').trim().split(/\s+/);
                if (hc && pts.length >= 3) {
                    ctx.beginPath();
                    pts.forEach(function(pt, i) {
                        var xy = pt.split(',');
                        if (i === 0) ctx.moveTo(parseFloat(xy[0]), parseFloat(xy[1]));
                        else         ctx.lineTo(parseFloat(xy[0]), parseFloat(xy[1]));
                    });
                    ctx.closePath();
                    ctx.fillStyle = hc;
                    ctx.fill();
                }
                [m.tTop, m.tBot].forEach(function(t) {
                    var fc = cs(t, 'fill');
                    if (!fc || !t.textContent) return;
                    var st = getComputedStyle(t);
                    ctx.setLineDash([]);
                    ctx.font         = st.fontStyle + ' ' + st.fontWeight + ' ' + st.fontSize + ' ' + st.fontFamily;
                    ctx.fillStyle    = fc;
                    ctx.textAlign    = 'center';
                    ctx.textBaseline = 'alphabetic';
                    ctx.fillText(t.textContent, n(t, 'x'), n(t, 'y'));
                });
            } catch (e) { console.warn('[screenshot] measurement draw failed', e); }
        });
        ctx.restore();
    }

    window.mcFsScreenshot = function() {
        if (_mcFsShotBusy) return;
        var overlay = document.getElementById('mc-fullscreen-overlay');
        if (!overlay || !overlay.classList.contains('open')) return;

        // Gate 1: the chart must be fully built for the symbol + timeframe currently on screen
        // (not "Loading…", "No data", "Failed to load", or mid-switch).
        if (!_mcFsChart || !_mcFsOhlcv || !_mcFsOhlcv.length || _mcFsBuiltSym !== _mcFsSym || _mcFsBuiltTf !== _mcFsTf) {
            _mcFsShotToast('Screenshot blocked: chart is not fully loaded', false);
            return;
        }
        if (!navigator.clipboard || !window.ClipboardItem) {
            _mcFsShotToast('Screenshot blocked: image clipboard needs https or localhost', false);
            return;
        }

        // Chart canvas at native pixel density: no crosshair, no upscaling.
        var shot;
        try { shot = _mcFsChart.takeScreenshot(true, false); } catch (e) { shot = null; }
        if (!shot || !shot.width || !shot.height) { _mcFsShotToast('Screenshot failed', false); return; }

        // Gate 2: refuse if the saved image would be too small to be sharp.
        if (shot.width < _MC_SHOT_MIN_WIDTH) {
            _mcFsShotToast('Screenshot blocked: only ' + shot.width + 'px wide (needs ' + _MC_SHOT_MIN_WIDTH + 'px+ for good quality)', false);
            return;
        }

        var container = document.getElementById('mc-fullscreen-chart');
        var hdr       = overlay.querySelector('.mc-fullscreen-header');
        var setRow    = document.getElementById('mc-fullscreen-settings');
        var mktInfo   = document.getElementById('mc-fs-mkt-info');
        var scale     = shot.width / (container.clientWidth || shot.width);   // canvas px per CSS px

        // Strip 1: header row, minus the Details / + / x controls.
        var hdrSkip = [];
        var detailsBtn = document.getElementById('mc-fullscreen-details-btn');
        var queueBtn   = document.getElementById('mc-fullscreen-queue-btn');
        var closeBtn   = hdr ? hdr.querySelector('.mc-fullscreen-close') : null;
        if (detailsBtn) { hdrSkip.push(detailsBtn); if (detailsBtn.previousElementSibling) hdrSkip.push(detailsBtn.previousElementSibling); }
        if (queueBtn) hdrSkip.push(queueBtn);
        if (closeBtn) hdrSkip.push(closeBtn);
        var hRect = hdr ? hdr.getBoundingClientRect() : null;

        // Fundamentals (EPS Q/Q ... MARGIN): on screen they sit left of the Details / + / x controls,
        // which are left out of the image, so they would stop short of the edge. Painted on their own
        // instead, shifted right so the last character ends at the header's right padding: the same
        // inset the symbol has on the left and the market info bar has on the right.
        // Screenshot only; the live layout is never touched.
        var fundEl = document.getElementById('mc-fullscreen-fund-stats');
        var fundShow = false, fundDx = 0;
        if (hdr && hRect && fundEl && getComputedStyle(fundEl).display !== 'none') {
            var fundRight = _mcFsShotInkRight(fundEl);
            if (fundRight !== null) {
                var padR = parseFloat(getComputedStyle(hdr).paddingRight) || 0;
                fundDx   = Math.max(0, (hRect.right - padR) - fundRight);
                fundShow = true;
                hdrSkip.push(fundEl);
            }
        }

        // Strip 2: the settings row's own background, holding ONLY the market info bar
        // (price/change, ADR, mkt cap, day range, 52W range) at its on-screen position.
        var sRect = null;
        if (setRow && mktInfo && getComputedStyle(mktInfo).display !== 'none') {
            sRect = setRow.getBoundingClientRect();
        }

        // Shared zoom for both strips, so the two rows stay in proportion to each other. Capped so that
        // (a) the left group (symbol, industry, badges) + a minimum gap + the fundamentals still fit
        // side by side in the header, and (b) the market info bar still fits in its row.
        var zoom = _MC_SHOT_STRIP_ZOOM;
        if (hdr && hRect && fundShow) {
            var leftRight = hRect.left;
            ['mc-fullscreen-sym', 'mc-fullscreen-meta', 'mc-fullscreen-rs-badge', 'mc-fullscreen-3mrs-badge'].forEach(function(id) {
                var el = document.getElementById(id);
                if (!el) return;
                var er = el.getBoundingClientRect();
                if (er.width && er.right > leftRight) leftRight = er.right;
            });
            var needHdr = (leftRight - hRect.left) + 16 + fundEl.getBoundingClientRect().width + (parseFloat(getComputedStyle(hdr).paddingRight) || 0);
            if (needHdr > 0) zoom = Math.min(zoom, hRect.width / needHdr);
        }
        if (sRect) {
            var setPad  = parseFloat(getComputedStyle(setRow).paddingLeft) || 0;
            var needSet = mktInfo.getBoundingClientRect().width + setPad * 2;
            if (needSet > 0) zoom = Math.min(zoom, sRect.width / needSet);
        }
        // Height: the two rows keep exactly the combined height they have on screen, so the image is no
        // taller than it was before the enlargement and never needs scrolling. The enlarged content is
        // centred in each row; the row padding shrinks to absorb the difference. Content height is measured
        // from what is actually painted, including the % labels that overflow above the range bars.
        function bbPx(el) {
            var cs = getComputedStyle(el), w = parseFloat(cs.borderBottomWidth) || 0;
            return (w > 0 && cs.borderBottomStyle !== 'none') ? Math.max(1, Math.round(w * scale)) : 0;
        }
        var hExt = null, sExt = null;
        if (hRect) {
            hExt = _mcFsShotExtentY(['mc-fullscreen-sym', 'mc-fullscreen-meta', 'mc-fullscreen-rs-badge',
                                     'mc-fullscreen-3mrs-badge', 'mc-fullscreen-fund-stats'].map(function(id) { return document.getElementById(id); }))
                   || { top: hRect.top, bottom: hRect.bottom };
        }
        if (sRect) sExt = _mcFsShotExtentY([mktInfo]) || { top: sRect.top, bottom: sRect.bottom };
        var nStrips = (hRect ? 1 : 0) + (sRect ? 1 : 0);
        var hdrPx0  = hRect ? Math.round(hRect.height * scale) : 0;
        var setPx0  = sRect ? Math.round(sRect.height * scale) : 0;
        var totPx   = hdrPx0 + setPx0;                       // combined on-screen height of the two rows
        var bbH = hRect ? bbPx(hdr) : 0, bbS = sRect ? bbPx(setRow) : 0;
        var hC  = hExt ? hExt.bottom - hExt.top : 0;         // content heights, CSS px
        var sC  = sExt ? sExt.bottom - sExt.top : 0;
        if (nStrips && hC + sC > 0) {
            zoom = Math.min(zoom, (totPx - bbH - bbS - 2 * nStrips * _MC_SHOT_STRIP_MIN_PAD * scale) / ((hC + sC) * scale));
        }
        zoom = Math.max(1, zoom);
        var zs = scale * zoom;     // canvas px per CSS px for the two top rows
        var rowPad = nStrips ? Math.max(0, (totPx - bbH - bbS - (hC + sC) * zs) / (2 * nStrips)) : 0;   // px above and below each row's content
        var hdrPx  = hRect ? Math.round(hC * zs + bbH + 2 * rowPad) : 0;
        var setPx  = sRect ? totPx - hdrPx : 0;
        var topPx  = hdrPx + setPx;                          // == totPx: the image is no taller than before
        console.debug('[screenshot] row zoom ' + zoom.toFixed(3) + ', row padding ' + rowPad.toFixed(1) + 'px, rows ' + hdrPx + '+' + setPx + 'px');

        var out = document.createElement('canvas');
        out.width  = shot.width;
        out.height = topPx + shot.height;
        var ctx = out.getContext('2d');
        ctx.fillStyle = themeColor('bg-page');
        ctx.fillRect(0, 0, out.width, out.height);
        ctx.drawImage(shot, 0, topPx);
        _mcFsShotPaintMeasures(ctx, _mcFsMeasureList, scale, topPx);   // SVG overlay: not part of the library's screenshot

        // A row's own background and bottom border, drawn crisp at exactly the row's pixel box.
        function paintBox(root, yPx, hPx) {
            var cs = getComputedStyle(root), bg = cs.backgroundColor, bb = bbPx(root);
            if (bg && bg !== 'transparent' && !/^rgba\(\s*\d+,\s*\d+,\s*\d+,\s*0\s*\)$/.test(bg)) {
                ctx.fillStyle = bg;
                ctx.fillRect(0, yPx, out.width, hPx);
            }
            if (bb) {
                ctx.fillStyle = cs.borderBottomColor;
                ctx.fillRect(0, yPx + hPx - bb, out.width, bb);
            }
        }
        // Draws the given elements at the zoomed size (zs), vertically centred in the row's content area
        // (midY = CSS y of the content's centre). anchorRight = false: the row's left edge maps to the
        // image's left edge, so left-aligned content keeps its (zoomed) inset from the left. true: the
        // row's right edge maps to the image's right edge, so right-aligned content keeps the same inset
        // from the right and can't run off the image. dx (optional) = extra horizontal shift in CSS px,
        // used to push the fundamentals to the right padding.
        function paintStrip(rect, yPx, hPx, bb, midY, roots, skip, dx, anchorRight) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, yPx, out.width, hPx - bb);
            ctx.clip();
            var tx = anchorRight ? out.width + ((dx || 0) - rect.right) * zs
                                 : (-rect.left + (dx || 0)) * zs;
            var ty = yPx + (hPx - bb) / 2 - midY * zs;
            ctx.setTransform(zs, 0, 0, zs, tx, ty);
            roots.forEach(function(r) {
                try { _mcFsShotPaint(ctx, r, skip, null); } catch (e) { console.warn('[screenshot] draw failed', e); }
            });
            ctx.restore();
        }
        if (hRect && hdrPx) {
            var hMid = (hExt.top + hExt.bottom) / 2;
            paintBox(hdr, 0, hdrPx);
            paintStrip(hRect, 0, hdrPx, bbH, hMid, Array.prototype.slice.call(hdr.children), hdrSkip, 0, false);
            if (fundShow) paintStrip(hRect, 0, hdrPx, bbH, hMid, [fundEl], [], fundDx, true);
        }
        if (sRect && setPx) {
            paintBox(setRow, hdrPx, setPx);
            paintStrip(sRect, hdrPx, setPx, bbS, (sExt.top + sExt.bottom) / 2, [mktInfo], [], 0, true);
        }

        _mcFsShotBusy = true;
        out.toBlob(function(blob) {
            if (!blob) { _mcFsShotBusy = false; _mcFsShotToast('Screenshot failed', false); return; }
            navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]).then(function() {
                _mcFsShotBusy = false;
                _mcFsShotToast('Screenshot copied (' + out.width + ' × ' + out.height + ')', true);
            }, function(err) {
                _mcFsShotBusy = false;
                _mcFsShotToast('Clipboard write failed' + (err && err.name ? ' (' + err.name + ')' : ''), false);
            });
        }, 'image/png');
    };

    function _buildFsChart(sym, ohlcv, tf) {
        var container = document.getElementById('mc-fullscreen-chart');
        container.innerHTML = '';
        if (_mcFsChart) { try { _mcFsChart.remove(); } catch(e) {} _mcFsChart = null; }
        _mcFsCandle = null; _mcFsVol = null; _mcFsVolMa = null; _mcFsVolData = null; _mcFsMaSeries = {}; _mcFsVwapSeries = []; _mcFsTrendlines = []; _mcFsTrendlineFirst = null;
        _mcFsTrendSvgOverlay = null; _mcFsTrendSvgLine = null; // SVG lives inside container.innerHTML = '' above
        _mcFsResetMeasure(); // committed measurements share the trendlines' lifetime (their DOM went with the container)
        _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
        _mcFsSelectedTrendlineIdx = -1;
        _mcFsSelectedVwapIdx = -1;
        _mcFsDismissCtx();

        _mcFsOhlcv    = ohlcv || [];
        _mcFsSym      = sym;
        _mcFsTf       = tf;
        _mcFsBuiltSym = sym;
        _mcFsBuiltTf  = tf;
        _mcFsLastCrosshairPrice = null;

        if (!window.LightweightCharts || !_mcFsOhlcv.length) {
            var _mcFsMsg = ohlcv === null ? 'Failed to load — click to retry' : 'No data';
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;' + (ohlcv === null ? 'cursor:pointer;text-decoration:underline;' : '') + '">' + _mcFsMsg + '</div>';
            if (ohlcv === null) {
                container.querySelector('div').addEventListener('click', function() {
                    fetchMcOhlcv(sym, tf).then(function(retryOhlcv) { _buildFsChart(sym, retryOhlcv, tf); });
                });
            }
            return;
        }

        // Register trendline mousedown in capture phase BEFORE createChart so our
        // listener fires before LW Charts' canvas handler and can stopPropagation.
        _mcFsTrendContRef = container;
        container.removeEventListener('mousedown', _onTrendMouseDown, true);
        container.addEventListener('mousedown', _onTrendMouseDown, true);

        // ── SVG overlay for lag-free trendline preview ─────────────────────
        // Reuse an existing overlay if the container already has one (e.g. after
        // a symbol reload), otherwise create a fresh one.
        var _existingSvg = container.querySelector('.mc-trend-svg-overlay');
        if (_existingSvg) {
            _mcFsTrendSvgOverlay = _existingSvg;
            _mcFsTrendSvgLine    = _existingSvg.querySelector('line');
        } else {
            _mcFsTrendSvgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            _mcFsTrendSvgOverlay.setAttribute('class', 'mc-trend-svg-overlay');
            _mcFsTrendSvgOverlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5;display:none;';
            _mcFsTrendSvgLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            _mcFsTrendSvgLine.setAttribute('stroke', _TRENDLINE_COLOR());
            _mcFsTrendSvgLine.setAttribute('stroke-width', '1.5');
            _mcFsTrendSvgLine.setAttribute('x1', '0'); _mcFsTrendSvgLine.setAttribute('y1', '0');
            _mcFsTrendSvgLine.setAttribute('x2', '0'); _mcFsTrendSvgLine.setAttribute('y2', '0');
            _mcFsTrendSvgOverlay.appendChild(_mcFsTrendSvgLine);
            container.style.position = 'relative';
            container.appendChild(_mcFsTrendSvgOverlay);
        }
        _mcFsTrendSvgOverlay.style.display = 'none'; // always start hidden on (re)load

        // ── Measure tool overlay ───────────────────────────────────────────
        var _mOver = _ensureMeasureOverlay(container, 'mc-fs-measure-svg', 'mc-fs-measure-info');
        _mcFsMeasureSvgOverlay = _mOver.svg;
        _mcFsMeasureSvgRect    = _mOver.rect;
        _mcFsMeasureHLine      = _mOver.hLine;
        _mcFsMeasureInfoDiv    = _mOver.info;
        _mcFsMeasureResult     = null; // clear stale result on chart rebuild
        _hideMeasureOverlay(_mcFsMeasureSvgOverlay, _mcFsMeasureInfoDiv);

        // mousemove on the container drives SVG line updates — no chart re-render
        container.removeEventListener('mousemove', _onTrendMouseMove);
        container.addEventListener('mousemove', _onTrendMouseMove);

        _mcFsChart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted'), panes: { separatorColor: themeColor('bg-subtle'), separatorHoverColor: themeColor('bg-surface-alpha') } },
            grid:    { vertLines: { visible: false }, horzLines: { visible: false } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted'), scaleMargins: { top: 0.05, bottom: 0.02 } },
            timeScale: { borderColor: themeColor('bg-surface'), timeVisible: false, secondsVisible: false, rightOffset: 24 },
            localization: { timeFormatter: _mcLwCrosshairDateFmt },
            handleScroll: true, handleScale: true,
        });
        _mcFsAttachCtxMenu(); // attach once, capture phase, safe to call repeatedly

        // Watermark — ticker + company name, bottom-right (top corners are
        // already occupied by the OHLC/MA legend and the pre/post badge).
        // Renders on the pane's background layer (candles draw on top of it
        // by design — LWC's own primitive renderer docs describe
        // drawBackground() as "usually watermarks"), so there's no fight
        // with price action for visibility. Company name comes from the meta
        // object already cached by fetchMcOhlcv for this exact symbol — no
        // extra request needed. New chart instance each rebuild means the
        // old watermark goes away with it; nothing to explicitly detach.
        var _fsMeta        = _mcMetaCache[sym] || {};
        var _fsCompanyName = _fsMeta.longName || _fsMeta.shortName || '';
        _mcFsWatermark = LightweightCharts.createTextWatermark(_mcFsChart.panes()[0], {
            horzAlign: 'right',
            vertAlign: 'bottom',
            lines: [
                { text: sym, color: themeColor('chart-watermark'), fontSize: _MC_WM_SYM_SIZE, fontStyle: _MC_WM_SYM_STYLE },
                _fsCompanyName ? { text: _fsCompanyName, color: themeColor('chart-watermark'), fontSize: _MC_WM_NAME_SIZE } : null,
            ].filter(Boolean),
        });

        _mcFsCandle = _mcFsChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'), borderVisible: false,
            wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
            priceLineVisible: false, lastValueVisible: true,
        });
        _mcFsCandle.setData(_mcFsOhlcv);

        _mcFsVol = _mcFsChart.addSeries(LightweightCharts.HistogramSeries, {
            color: themeColor('al-chart-volume'), priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: true,
        }, 1);
        _mcFsVol.setData(_mcFsOhlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
        }));
        _mcFsVol.priceScale().applyOptions({
            visible: true,
            borderColor: themeColor('bg-surface'),
            textColor: themeColor('text-muted'),
            minimumWidth: 60,
        });

        // 50 SMA on volume — plotted in the same volume pane (pane 1)
        (function() {
            var period = 50;
            _mcFsVolData = [];
            for (var i = period - 1; i < _mcFsOhlcv.length; i++) {
                var sum = 0;
                for (var j = i - (period - 1); j <= i; j++) sum += (_mcFsOhlcv[j].volume || 0);
                _mcFsVolData.push({ time: _mcFsOhlcv[i].time, value: sum / period });
            }
            _mcFsVolMa = _mcFsChart.addSeries(LightweightCharts.LineSeries, {
                color: '#1848cc', lineWidth: 1,
                priceLineVisible: false, lastValueVisible: true,
                crosshairMarkerVisible: false,
            }, 1);
            _mcFsVolMa.setData(_mcFsVolData);
        })();
        _mcFsVolSmaMap = _mcFsVolData && _mcFsVolData.length
            ? new Map(_mcFsVolData.map(function(d) { return [d.time, d.value]; }))
            : null;

        // Pin volume pane to ~22% of chart height so price pane fills the rest
        (function() {
            var panes = _mcFsChart.panes();
            if (panes && panes.length >= 2) {
                var totalH = container ? container.offsetHeight : 700;
                panes[1].setHeight(Math.round(totalH * 0.22));
            }
        })();

        // Vol % vs 50-SMA label — tracks last volume bar on scroll/zoom
        (function() {
            if (!_mcFsVolData || !_mcFsVolData.length || !_mcFsOhlcv.length) return;
            var lastBar = _mcFsOhlcv[_mcFsOhlcv.length - 1];
            var lastVol = lastBar.volume;
            var sma50   = _mcFsVolData[_mcFsVolData.length - 1].value;
            if (!sma50) return;

            // DST-aware ET market window
            function nthSunday(yr, mo, n) {
                var d = new Date(Date.UTC(yr, mo, 1));
                return new Date(Date.UTC(yr, mo, 1 + (7 - d.getUTCDay()) % 7 + (n - 1) * 7));
            }
            var now     = Date.now();
            var barDate = new Date(lastBar.time * 1000);
            var yr = barDate.getUTCFullYear(), mo = barDate.getUTCMonth(), dy = barDate.getUTCDate();
            var isDST   = barDate >= nthSunday(yr, 2, 2) && barDate < nthSunday(yr, 10, 1);
            var etDelta = isDST ? 4 : 5;                           // EDT = UTC-4 | EST = UTC-5
            var mktOpen  = new Date(Date.UTC(yr, mo, dy,  9 + etDelta, 30)); // 09:30 ET
            var mktClose = new Date(Date.UTC(yr, mo, dy, 16 + etDelta,  0)); // 16:00 ET
            var totalMs  = mktClose - mktOpen;
            var timeratio = 1.0;
            if (now > mktOpen && now < mktClose) timeratio = totalMs / (now - mktOpen);
            var projectedVol = lastVol * timeratio;
            var volDiffPct   = (projectedVol / sma50 - 1) * 100;

            var sign  = volDiffPct >= 0 ? '+' : '';
            var color = volDiffPct >= 0 ? 'var(--success)' : 'var(--danger)';

            var lbl = document.createElement('div');
            lbl.id = 'mc-fs-vol-pct-label';
            lbl.style.cssText = 'position:absolute;z-index:20;pointer-events:none;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:3px;white-space:nowrap;line-height:1;';
            lbl.innerHTML = '<span style="color:var(--border-muted);">›</span>'
                          + '<span style="color:' + color + ';">' + sign + volDiffPct.toFixed(1) + '%</span>';
            container.appendChild(lbl);

            // Resolve volume pane Y once after first render (pane height doesn't change on scroll)
            setTimeout(function() {
                if (!_mcFsChart) return;

                var volPaneTop = 0, volPaneH = 0;
                try {
                    var panes = _mcFsChart.panes();
                    var pe = (panes && panes[1] && typeof panes[1].getElement === 'function')
                             ? panes[1].getElement() : null;
                    if (pe) {
                        var r = pe.getBoundingClientRect();
                        var cr = container.getBoundingClientRect();
                        volPaneTop = r.top  - cr.top;
                        volPaneH   = r.height;
                    }
                } catch(e) {}

                if (!volPaneH) {
                    var totalH = container.offsetHeight;
                    volPaneH   = Math.round(totalH * 0.22);
                    volPaneTop = totalH - volPaneH - 22; // ~22px time axis
                }

                var lblTop = (volPaneTop + volPaneH - 28) + 'px';

                // Reposition on every scroll / zoom — dies automatically when lbl is detached
                function positionVolLabel() {
                    if (!lbl.isConnected || !_mcFsChart) return;
                    var lastX = _mcFsChart.timeScale().timeToCoordinate(lastBar.time);
                    // Plot area = container minus the right price scale. Anything past
                    // that bleeds out of the chart (this label is a DOM div, not canvas).
                    var scaleW = 0;
                    try { scaleW = _mcFsChart.priceScale('right').width(); } catch (e) {}
                    var plotW = container.clientWidth - scaleW;
                    lbl.style.display = 'flex';          // must be visible to measure
                    if (lastX == null || lastX < 0 || lastX + 10 + lbl.offsetWidth > plotW) {
                        lbl.style.display = 'none';
                        return;
                    }
                    lbl.style.left = (lastX + 10) + 'px';
                    lbl.style.top  = lblTop;
                }

                positionVolLabel(); // initial paint
                // Logical range fires on every pan/zoom step; the time-range event only
                // fires when the first/last visible bar changes, so the label lagged.
                _mcFsChart.timeScale().subscribeVisibleLogicalRangeChange(positionVolLabel);
            }, 60);
        })();

        // Re-render measure overlay on pan/zoom so the rect tracks correctly
        _mcFsChart.timeScale().subscribeVisibleLogicalRangeChange(function() {
            if (_mcFsMeasureResult) {
                _renderMeasureOverlay(_mcFsChart, _mcFsCandle, _mcFsTrendContRef,
                    _mcFsMeasureSvgOverlay, _mcFsMeasureSvgRect, _mcFsMeasureHLine,
                    _mcFsMeasureInfoDiv, _mcFsMeasureResult);
            }
            _measureRenderAll(_mcFsMeasureList, _mcFsChart, _mcFsCandle, _mcFsTrendContRef, _mcFsOhlcv);
        });

        // Active MAs
        Object.keys(_mcFsActiveMas).forEach(function(key) {
            if (!_mcFsActiveMas[key]) return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = _mcFsChart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
            var maData = _calcMA(_mcFsOhlcv, key);
            s.setData(maData);
            _mcFsMaSeries[key]  = s;
            _mcFsMaDataMap[key] = new Map(maData.map(function(d) { return [d.time, d.value]; }));
        });

        // Visible range
        var n = _mcFsOhlcv.length;
        _mcFsChart.timeScale().setVisibleLogicalRange({ from: n - _mcFsVisibleBars, to: n + 15 });

        // Click handler — AVWAP anchor + AVWAP line selection
        _mcFsChart.subscribeClick(function(param) {
            // ── AVWAP anchor ───────────────────────────────────────────────
            if (_mcFsVwapMode) {
                if (!param.time) return;
                var idx = _barIdxByTime(_mcFsOhlcv, param.time);
                if (idx < 0) return;
                var _nVwBefore = _mcFsVwapSeries.length;
                _addFsVwap(idx);
                if (_mcFsVwapSeries.length > _nVwBefore) {
                    _cdAddAv(_mcFsSym, _mcFsOhlcv, _mcFsTf, idx);   // save the anchor
                    // Select the new AVWAP so Delete removes it straight away. Trendlines are deselected first:
                    // the key handler checks them first and would otherwise delete an older selected line instead.
                    _deselectAllTrendlines();
                    _selectVwap(_mcFsVwapSeries.length - 1);
                }
                return;
            }
            // Don't interfere with trendline tool
            if (_mcFsTrendlineMode) return;
            // ── AVWAP line selection ───────────────────────────────────────
            if (!_mcFsVwapSeries.length || !param.time || !param.point) {
                if (_mcFsSelectedVwapIdx !== -1) _deselectAllVwaps();
                return;
            }
            var HIT_PX = 8;
            var hitIdx = -1;
            _mcFsVwapSeries.forEach(function(entry, i) {
                if (!entry.dataMap) return;
                var avwapVal = entry.dataMap.get(param.time);
                if (avwapVal == null) return;
                var yCoord = entry.series.priceToCoordinate(avwapVal);
                if (yCoord == null) return;
                if (Math.abs(param.point.y - yCoord) <= HIT_PX) hitIdx = i;
            });
            if (hitIdx !== -1) {
                if (_mcFsSelectedVwapIdx === hitIdx) {
                    _deselectAllVwaps();
                } else {
                    _selectVwap(hitIdx);
                }
            } else {
                if (_mcFsSelectedVwapIdx !== -1) _deselectAllVwaps();
            }
        });

        // OHLC legend
        var leg = document.createElement('div');
        leg.id = 'mc-fs-legend';
        leg.style.cssText = 'position:absolute;top:8px;left:14px;z-index:10;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-muted-2);pointer-events:none;line-height:1.8;background:var(--bg-page-alpha-3);padding:4px 10px;border-radius:4px;';
        container.style.position = 'relative';
        container.appendChild(leg);

        // Pre/post-market price badge — top-right, hidden unless meta says we're
        // in an extended session and Yahoo actually gave us a price for it.
        var prepost = document.createElement('div');
        prepost.id = 'mc-fs-prepost-badge';
        prepost.style.cssText = 'position:absolute;top:8px;right:8px;z-index:10;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;pointer-events:none;line-height:1.6;padding:3px 8px;border-radius:4px;display:none;';
        container.appendChild(prepost);

        function fp(v) { return v != null ? v.toFixed(2) : '—'; }
        function fv(v) { return v==null?'—':v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':v.toFixed(0); }

        _mcFsChart.subscribeCrosshairMove(function(p) {
            // Always track the real cursor y-position as the alert price.
            // d.close would stay stale at the last candle when the crosshair
            // moves into empty space — p.point.y gives the exact horizontal
            // line position regardless of whether a candle is under the cursor.
            if (p.point && _mcFsCandle) {
                var cursorPrice = _mcFsCandle.coordinateToPrice(p.point.y);
                _mcFsLastCrosshairPrice = (cursorPrice != null && !isNaN(cursorPrice)) ? cursorPrice : null;
            } else {
                _mcFsLastCrosshairPrice = null;
            }
            // Track bar time for MA proximity detection on right-click
            _mcFsLastCrosshairTime = p.time || null;
            if (!p.time || !p.seriesData || !p.seriesData.size) {
                leg.innerHTML = '';
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
                return;
            }
            var d = p.seriesData.get(_mcFsCandle);
            if (!d) {
                leg.innerHTML = '';
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
                return;
            }
            var cl = d.close >= d.open ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
            var vd = p.seriesData.get(_mcFsVol);
            // Price change from previous candle
            var chgHtml = '';
            var barIdx = _barIdxByTime(_mcFsOhlcv, p.time);
            if (barIdx > 0) {
                var prevClose = _mcFsOhlcv[barIdx - 1].close;
                var delta = d.close - prevClose;
                var pct = (delta / prevClose) * 100;
                var chgClr = delta >= 0 ? 'var(--success)' : 'var(--danger)';
                chgHtml = '&nbsp;&nbsp;<span style="color:' + chgClr + '">'
                        + (delta >= 0 ? '+' : '') + delta.toFixed(2)
                        + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)'
                        + '</span>';
            }
            leg.innerHTML =
                '<span style="color:var(--text-muted-2)">O</span> <span style="color:'+cl+'">'+fp(d.open)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">H</span> <span style="color:'+cl+'">'+fp(d.high)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">L</span> <span style="color:'+cl+'">'+fp(d.low)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">C</span> <span style="color:'+cl+'">'+fp(d.close)+'</span>' +
                chgHtml +
                (vd ? '&nbsp;&nbsp;<span style="color:var(--text-muted)">V</span> <span style="color:var(--text-muted-2)">'+fv(vd.value)+'</span>' : '');
            // Floating tooltip
            if (_mcFsTooltipEnabled) {
                var ttDiv = _getLwTooltipDiv();
                ttDiv.innerHTML = _buildTooltipHtml(d, barIdx, _mcFsOhlcv, _mcFsVolSmaMap, _mcFsMaDataMap, _mcFsActiveMas, p.time);
                ttDiv.style.display = 'block';
                if (p.point) {
                    var rect = container.getBoundingClientRect();
                    _positionTooltip(ttDiv, rect.left + p.point.x, rect.top + p.point.y, rect.right);
                }
            } else if (_lwTooltipDiv) {
                _lwTooltipDiv.style.display = 'none';
            }
        });

        // ── Market info strip (price/change, day range, 52W range) ──────────────
        (function() {
            var n = _mcFsOhlcv.length;
            if (!n) return;
            var last   = _mcFsOhlcv[n - 1];
            var prev   = n > 1 ? _mcFsOhlcv[n - 2] : null;
            var close  = last.close;
            var chg    = prev ? close - prev.close : 0;
            var pct    = prev ? chg / prev.close * 100 : 0;
            var dayLow = last.low, dayHigh = last.high;

            // 52W range — lookback adjusted per timeframe
            var yrBars = _mcFsTf === 'W' ? 52 : _mcFsTf === 'M' ? 12 : 252;
            var slice  = _mcFsOhlcv.slice(-Math.min(yrBars, n));
            var yrLow  = slice.reduce(function(m, b) { return Math.min(m, b.low);  }, Infinity);
            var yrHigh = slice.reduce(function(m, b) { return Math.max(m, b.high); }, -Infinity);

            var chgColor = chg >= 0 ? 'var(--success)' : 'var(--danger)';
            var chgSign  = chg >= 0 ? '+' : '';
            var barLabel = _mcFsTf === 'W' ? 'WK' : _mcFsTf === 'M' ? 'MO' : 'DAY';

            // Gradient range bar: red→yellow→green track, dark overlay masks unfilled right,
            // white dot with dark ring marks current price position
            var barColor = chg >= 0 ? 'var(--al-chart-up)' : 'var(--al-chart-down)';

            // Shared bar builder — 4px tall, matches 52W style
            function mkBar(low, high, curr, width, crLabel) {
                var pos = (high > low)
                    ? Math.max(2, Math.min(98, (curr - low) / (high - low) * 100))
                    : 50;
                var p = pos.toFixed(1);
                var crSpan = crLabel != null
                    ? '<span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-150%);' +
                      'font-size:9px;font-weight:700;color:' + crLabel.color + ';letter-spacing:.02em;pointer-events:none;">' +
                      crLabel.text + '</span>'
                    : '';
                return '<span style="position:relative;display:inline-block;width:' + width + 'px;height:4px;' +
                    'border-radius:2px;background:var(--bg-surface);vertical-align:middle;flex-shrink:0;overflow:visible;">' +
                    '<span style="position:absolute;left:0;top:0;height:100%;width:' + p + '%;background:' + barColor + ';border-radius:2px;"></span>' +
                    '<span style="position:absolute;top:50%;left:' + p + '%;' +
                    'transform:translate(-50%,-50%);width:8px;height:8px;' +
                    'background:var(--text-primary-alt);border-radius:50%;box-shadow:0 0 0 1.5px var(--bg-page);"></span>' +
                    crSpan +
                    '</span>';
            }

            // CR% value computed live from day range
            var crRaw   = (dayHigh > dayLow) ? Math.round((close - dayLow) / (dayHigh - dayLow) * 100) : null;
            var crLabel = crRaw != null ? {
                text:  crRaw + '%',
                color: crRaw >= 60 ? 'var(--success)' : crRaw >= 30 ? 'var(--warning-alt)' : 'var(--danger)'
            } : null;

            var adrEl = document.getElementById('mc-fs-mkt-adr');
            var sd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
            if (adrEl) {
                var adrRaw = sd ? sd.adr_pct : null;
                if (adrRaw != null) {
                    adrEl.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">ADR%</span>'
                                    + '<span style="color:var(--text-primary-alt);font-size:12px;">' + adrRaw.toFixed(1) + '%</span>';
                    adrEl.style.display = 'inline-flex';
                } else {
                    adrEl.style.display = 'none';
                }
            }

            var mcapEl = document.getElementById('mc-fs-mkt-mcap');
            if (mcapEl) {
                var mcapRaw = sd ? sd.MarketCap : null;
                if (mcapRaw != null) {
                    var mc = mcapRaw >= 1e12 ? (mcapRaw/1e12).toFixed(2)+'T'
                           : mcapRaw >= 1e9  ? (mcapRaw/1e9).toFixed(2)+'B'
                           : mcapRaw >= 1e6  ? (mcapRaw/1e6).toFixed(0)+'M' : mcapRaw;
                    mcapEl.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">Mkt Cap</span>'
                                     + '<span style="color:var(--text-primary-alt);font-size:12px;">' + mc + '</span>';
                    mcapEl.style.display = 'inline-flex';
                } else {
                    mcapEl.style.display = 'none';
                }
            }

            document.getElementById('mc-fs-mkt-price').innerHTML =
                '<span style="color:var(--text-emphasis-2);font-size:20px;font-weight:700;">' + fp(close) + '</span>' +
                '&nbsp;<span style="color:' + chgColor + ';font-size:13px;font-weight:600;">' +
                chgSign + fp(chg) + '&nbsp;(' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)</span>';

            document.getElementById('mc-fs-mkt-day').innerHTML =
                '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">' + barLabel + '</span>' +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(dayLow) + '</span>' +
                mkBar(dayLow, dayHigh, close, 130, crLabel) +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(dayHigh) + '</span>';

            var w52HiPct   = (yrHigh > 0) ? (yrHigh - close) / yrHigh * 100 : 0;
            var w52HiLabel = yrHigh > 0 ? {
                text:  w52HiPct < 0.5 ? 'ATH' : ('-' + w52HiPct.toFixed(1) + '%'),
                color: w52HiPct <= 5 ? 'var(--success)' : w52HiPct <= 15 ? 'var(--warning-alt)' : 'var(--danger)'
            } : null;
            document.getElementById('mc-fs-mkt-52w').innerHTML =
                '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">52W</span>' +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(yrLow) + '</span>' +
                mkBar(yrLow, yrHigh, close, 120, w52HiLabel) +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(yrHigh) + '</span>';

            document.getElementById('mc-fs-mkt-info').style.display = 'flex';
        })();

        // ── Pre/post-market badge content ────────────────────────────────────
        // NOTE: this can no longer be read out of _mcMetaCache[sym] — Yahoo's
        // chart-meta (what _mcMetaCache holds) never includes marketState or
        // extended-hours prices. It's fetched separately via a dedicated Worker
        // route; see _mcFsRenderPrePostBadge below.
        _mcFsRenderPrePostBadge(sym);

        // Delete/Escape key handler — trendlines + AVWAP
        if (_mcFsKeyHandler) { document.removeEventListener('keydown', _mcFsKeyHandler); }
        _mcFsKeyHandler = function(evt) {
            if (
                evt.key.length === 1 && /[a-zA-Z0-9]/.test(evt.key) &&
                !evt.ctrlKey && !evt.metaKey && !evt.altKey &&
                evt.target.tagName !== 'INPUT' && evt.target.tagName !== 'TEXTAREA' &&
                !document.getElementById('mc-fs-sym-input')
            ) {
                window._mcFsSymClick();
                var _quickInp = document.getElementById('mc-fs-sym-input');
                if (_quickInp) {
                    _quickInp.value = evt.key.toUpperCase();
                    _quickInp.dispatchEvent(new Event('input'));
                }
                evt.preventDefault();
                return;
            }
            // Escape: cancel in-progress draw OR deselect selected trendline OR clear measure
            if (evt.key === 'Escape') {
                if (_mcFsMeasureActive || _mcFsMeasurePhase === 1) {
                    _mcFsMeasureActive = false;
                    _mcFsMeasurePhase  = 0;
                    if (_mcFsMeasureRafId) { cancelAnimationFrame(_mcFsMeasureRafId); _mcFsMeasureRafId = null; }
                    document.removeEventListener('mousemove', _onMcFsMeasureDragMove);
                    document.removeEventListener('mouseup',   _onMcFsMeasureDragEnd);
                    document.removeEventListener('mousemove', _onMcFsMeasurePreviewMove);
                }
                if (_mcFsMeasureResult) {
                    _hideMeasureOverlay(_mcFsMeasureSvgOverlay, _mcFsMeasureInfoDiv);
                    _mcFsMeasureResult = null;
                }
                if (_mcFsTrendDraw.active) {
                    _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
                    if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
                } else if (_mcFsSelectedTrendlineIdx !== -1) {
                    _deselectAllTrendlines();
                } else if (_mcFsSelectedVwapIdx !== -1) {
                    _deselectAllVwaps();
                }
                return;
            }
            // Alt shortcuts: D = tooltip, T = trendline, A = AVWAP, C = copy symbol, S = screenshot
            if (evt.altKey && !evt.ctrlKey && !evt.metaKey) {
                if (evt.key === 'd' || evt.key === 'D') { evt.preventDefault(); window.mcFsToggleTooltip(); return; }
                if (evt.key === 't' || evt.key === 'T') { evt.preventDefault(); window.mcFsToggleTrendline(); return; }
                if (evt.key === 'a' || evt.key === 'A') { evt.preventDefault(); window.mcFsToggleVwap(); return; }
                // Alt+S: copy a hi-res PNG of the header + chart to the clipboard. Same input
                // guards as Alt+C so typing in the symbol box is never hijacked. evt.code is
                // checked too because Option+S on a Mac produces a different evt.key.
                if ((evt.key === 's' || evt.key === 'S' || evt.code === 'KeyS') &&
                    evt.target.tagName !== 'INPUT' && evt.target.tagName !== 'TEXTAREA' &&
                    !document.getElementById('mc-fs-sym-input')) {
                    evt.preventDefault(); window.mcFsScreenshot(); return;
                }
                // Alt+C: same as clicking the header's copy-symbol button. Skipped while the
                // symbol search box (or any input) has focus so typing is never hijacked.
                if ((evt.key === 'c' || evt.key === 'C' || evt.code === 'KeyC') &&
                    evt.target.tagName !== 'INPUT' && evt.target.tagName !== 'TEXTAREA' &&
                    !document.getElementById('mc-fs-sym-input')) {
                    var _qBtn = document.getElementById('mc-fullscreen-queue-btn');
                    if (_qBtn) { evt.preventDefault(); _qBtn.click(); return; }
                }
            }
            if (evt.key !== 'Delete') return;
            // Don't steal Delete from the symbol input
            if (document.getElementById('mc-fs-sym-input')) return;
            // Delete a selected measurement first
            if (_measureDeleteSelected(_mcFsMeasureList)) { evt.preventDefault(); return; }
            // Delete selected trendline first (takes priority over "delete last")
            if (_mcFsSelectedTrendlineIdx !== -1) {
                var selTl = _mcFsTrendlines[_mcFsSelectedTrendlineIdx];
                _mcFsSelectedTrendlineIdx = -1;
                if (selTl) _deleteTrendlineWithAlerts(_mcFsSym, selTl, function() {
                    var ti = _mcFsTrendlines.indexOf(selTl);
                    if (ti !== -1) _mcFsTrendlines.splice(ti, 1);
                    try { if (_mcFsCandle) _mcFsCandle.detachPrimitive(selTl.primitive); } catch(e) {}
                });
                return;
            }
            // Delete selected AVWAP
            if (_mcFsSelectedVwapIdx !== -1) {
                var selVwap = _mcFsVwapSeries[_mcFsSelectedVwapIdx];
                _mcFsSelectedVwapIdx = -1;
                if (selVwap) _deleteVwapWithAlerts(_mcFsSym, _mcFsOhlcv, _mcFsTf, selVwap, function() {
                    var vi = _mcFsVwapSeries.indexOf(selVwap);
                    if (vi !== -1) _mcFsVwapSeries.splice(vi, 1);
                    try { _mcFsChart.removeSeries(selVwap.series); } catch(e) {}
                    _mcFsVwapSeries.forEach(function(entry) { _vwapSetSelectedLook(entry, false); });
                });
                return;
            }
            // Trendline delete (last) when draw tool is active
            if (_mcFsTrendlineMode && _mcFsTrendlines.length) {
                var tLast = _mcFsTrendlines[_mcFsTrendlines.length - 1];
                _deleteTrendlineWithAlerts(_mcFsSym, tLast, function() {
                    var li = _mcFsTrendlines.indexOf(tLast);
                    if (li !== -1) _mcFsTrendlines.splice(li, 1);
                    try { if (_mcFsCandle) _mcFsCandle.detachPrimitive(tLast.primitive); } catch(e) {}
                });
                return;
            }
        };
        document.addEventListener('keydown', _mcFsKeyHandler);

        // Tooltip button (injected once, idempotent)
        (function() {
            var avwapBtn = document.getElementById('mc-fs-vwap-btn');
            if (avwapBtn && !document.getElementById('mc-fs-tooltip-btn')) {
                var ttBtn = document.createElement('button');
                ttBtn.id        = 'mc-fs-tooltip-btn';
                ttBtn.className = avwapBtn.className.replace(/\bactive\b/g, '').trim();
                ttBtn.title     = 'Data Tooltip (Alt+D)';
                ttBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><line x1="6" y1="1" x2="6" y2="11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><line x1="1" y1="6" x2="11" y2="6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
                ttBtn.addEventListener('click', window.mcFsToggleTooltip);
                avwapBtn.parentNode.insertBefore(ttBtn, avwapBtn.nextSibling);
            }
            var existing = document.getElementById('mc-fs-tooltip-btn');
            if (existing) existing.classList.toggle('active', _mcFsTooltipEnabled);
        })();

        // Inject today's live bar into the fullscreen chart so the latest
        // intraday OHLC is always reflected, even if Yahoo's historical feed
        // returned a stale or missing current-day bar.
        _injectChartLiveBar(sym, tf, _mcFsCandle, _mcFsVol, _mcFsOhlcv,
            function() { return _mcFsSym !== sym || !_mcFsCandle; });

        // Keep it updating for as long as this chart stays open — see
        // _mcFsStartLiveTick for why this can't just reuse the caches above.
        _mcFsStartLiveTick(sym, tf);

        // Restore alert-backed trendlines and AVWAPs so they're visible when reviewing the chart
        _restoreAlertLines(sym, tf, ohlcv, _addFsTrendline, _addFsVwap);
        // Then the lines you drew by hand. After the alert-backed ones, so a line that also backs an alert isn't doubled.
        (function() {
            var chartRef = _mcFsChart;
            _cdRestore(sym, tf, {
                isStale:        function() { return _mcFsChart !== chartRef || _mcFsSym !== sym || !_mcFsCandle; },
                getOhlcv:       function() { return _mcFsOhlcv; },
                getTrendlines:  function() { return _mcFsTrendlines; },
                getVwapAnchors: function() { return _mcFsVwapSeries.map(function(v) { return v.anchor; }); },
                addTrendline:   _addFsTrendline,
                addVwap:        _addFsVwap
            });
        })();
        // The measurements you committed on this ticker (by time + price, so any timeframe).
        (function() {
            var chartRef = _mcFsChart;
            _cdRestoreMs(sym, {
                isStale:     function() { return _mcFsChart !== chartRef || _mcFsSym !== sym || !_mcFsCandle; },
                getOhlcv:    function() { return _mcFsOhlcv; },
                getMeasures: function() { return _mcFsMeasureList; },
                addMeasure:  function(a, b) {
                    var ms = _measureCommit({ contRef: _mcFsTrendContRef, measureList: _mcFsMeasureList }, a.time, a.price, b.time, b.price);
                    ms.sym = sym;
                },
                render:      function() { _measureRenderAll(_mcFsMeasureList, _mcFsChart, _mcFsCandle, _mcFsTrendContRef, _mcFsOhlcv); }
            });
        })();
    }

    // Fullscreen window-level controls
    window.mcFsSetTf = function(tf) {
        if (!_mcFsSym) return;
        _mcFsTf = tf;
        document.querySelectorAll('.mc-fs-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        // Reset AVWAP
        _mcFsVwapMode = false; _mcFsVwapSeries = []; _mcFsSelectedVwapIdx = -1;
        var vwapBtn  = document.getElementById('mc-fs-vwap-btn');
        if (vwapBtn)  vwapBtn.classList.remove('active');
        // Reset trendlines (symbol stays same but data reloads)
        _mcFsTrendlines = []; _mcFsTrendlineFirst = null;
        _mcFsSelectedTrendlineIdx = -1;
        if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
        _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
        var tBtn  = document.getElementById('mc-fs-trendline-btn');
        // Reset measure tool
        _mcFsResetMeasure();
        _mcFsMeasureMode = false; _mcFsMeasureActive = false; _mcFsMeasurePhase = 0; _mcFsMeasureResult = null;
        if (_mcFsMeasureRafId) { cancelAnimationFrame(_mcFsMeasureRafId); _mcFsMeasureRafId = null; }
        var mBtn = document.getElementById('mc-fs-measure-btn');
        if (mBtn) mBtn.classList.remove('active');
        document.removeEventListener('mousemove', _onMcFsMeasureDragMove);
        document.removeEventListener('mouseup',   _onMcFsMeasureDragEnd);
        document.removeEventListener('mousemove', _onMcFsMeasurePreviewMove);
        // Close MA panel
        var maPanel   = document.getElementById('mc-fs-ma-panel');
        var maChevron = document.getElementById('mc-fs-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
        // Default viewport per TF
        _mcFsVisibleBars = tf === 'D' ? 252 : tf === 'W' ? 104 : 60;
        // No forced cache-clear here anymore — if this TF was already fetched
        // this session, fetchMcOhlcv's cache-hit path serves it instantly with
        // no network round-trip. Today's price still lands via the separate
        // _injectChartLiveBar call inside _buildFsChart either way, so this
        // isn't trading away freshness, just an unconditional re-fetch that
        // wasn't buying anything on top of that.
        // Fetch + rebuild
        var sym = _mcFsSym;
        var container = document.getElementById('mc-fullscreen-chart');
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;">Loading\u2026</div>';
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (!document.getElementById('mc-fullscreen-overlay').classList.contains('open')) return;
            if (_mcFsSym !== sym || _mcFsTf !== tf) return;
            _buildFsChart(sym, ohlcv, tf);
        });
    };

    function _mcFsUpdateMaBadge() {
        // no-op: MA button stays neutral regardless of active MA count
    }

    window.mcFsToggleMaPanel = function(e) {
        e.stopPropagation();
        var panel   = document.getElementById('mc-fs-ma-panel');
        var chevron = document.getElementById('mc-fs-ma-chevron');
        if (!panel) return;
        var opening = panel.style.display === 'none';
        panel.style.display = opening ? '' : 'none';
        if (chevron) chevron.style.transform = opening ? 'rotate(180deg)' : '';
        if (opening) {
            setTimeout(function() {
                function _outsideClick(ev) {
                    var wrap = document.getElementById('mc-fs-ma-wrap');
                    if (wrap && !wrap.contains(ev.target)) {
                        panel.style.display = 'none';
                        if (chevron) chevron.style.transform = '';
                        document.removeEventListener('click', _outsideClick, true);
                    }
                }
                document.addEventListener('click', _outsideClick, true);
            }, 0);
        }
    };

    window.mcFsToggleMa = function(key) {
        _mcFsActiveMas[key] = !_mcFsActiveMas[key];
        var btn = document.getElementById('mc-fs-ma-' + key);
        if (btn) btn.classList.toggle('active', _mcFsActiveMas[key]);
        _mcFsUpdateMaBadge();
        if (!_mcFsChart || !_mcFsOhlcv.length) return;
        if (_mcFsActiveMas[key]) {
            if (_mcFsMaSeries[key]) return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = _mcFsChart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
            var maData = _calcMA(_mcFsOhlcv, key);
            s.setData(maData);
            _mcFsMaSeries[key]  = s;
            _mcFsMaDataMap[key] = new Map(maData.map(function(d) { return [d.time, d.value]; }));
        } else {
            if (_mcFsMaSeries[key]) { try { _mcFsChart.removeSeries(_mcFsMaSeries[key]); } catch(e) {} delete _mcFsMaSeries[key]; }
            delete _mcFsMaDataMap[key];
        }
    };

    window.mcFsToggleVwap = function() {
        _mcFsVwapMode = !_mcFsVwapMode;
        var btn  = document.getElementById('mc-fs-vwap-btn');
        if (btn) btn.classList.toggle('active', _mcFsVwapMode);
        // Deactivate trendline tool if AVWAP is being turned on
        if (_mcFsVwapMode && _mcFsTrendlineMode) {
            _mcFsTrendlineMode = false;
            var tBtn = document.getElementById('mc-fs-trendline-btn');
            if (tBtn) tBtn.classList.remove('active');
            _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
            if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
        }
        if (_mcFsVwapMode && _mcFsMeasureMode) {
            _mcFsMeasureMode = false;
            var mBtn = document.getElementById('mc-fs-measure-btn');
            if (mBtn) mBtn.classList.remove('active');
        }
    };

    window.mcFsToggleTrendline = function() {
        _mcFsTlMenu.resetStyle(); // plain click and Alt+T always draw solid
        _mcFsTrendlineMode = !_mcFsTrendlineMode;
        var btn  = document.getElementById('mc-fs-trendline-btn');
        if (btn) btn.classList.toggle('active', _mcFsTrendlineMode);
        // Deactivate AVWAP tool if trendline is being turned on
        if (_mcFsTrendlineMode && _mcFsVwapMode) {
            _mcFsVwapMode = false;
            var vBtn = document.getElementById('mc-fs-vwap-btn');
            if (vBtn) vBtn.classList.remove('active');
        }
        if (_mcFsTrendlineMode && _mcFsMeasureMode) {
            _mcFsMeasureMode = false;
            var mBtn = document.getElementById('mc-fs-measure-btn');
            if (mBtn) mBtn.classList.remove('active');
        }
        // Cancel any in-progress draw and deselect
        _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
        _mcFsTrendlineFirst = null;
        if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
        if (_mcFsSelectedTrendlineIdx !== -1) _deselectAllTrendlines();
    };


    // ── Trendline button: press and hold → pick Solid or Dotted ─────────────
    // ONE implementation shared by the fullscreen, watchlist and alerts charts: each builds its own instance (below,
    // and in alerts.js), so all three behave identically and a change here applies to all of them.
    // A normal click (or Alt+T) still just toggles the tool and draws solid. Holding the button for HOLD_MS opens a
    // small menu under it; release over an option to arm the tool with that style, release anywhere else to cancel.
    // After the line is drawn the tool switches itself off as before, so a dotted line needs the hold again.
    //   cfg.toggle()      the chart's own trendline toggle (what a plain click / Alt+T runs). It must call resetStyle().
    //   cfg.isActive()    true while the trendline tool is armed
    //   cfg.cancelDraw()  drop a half-drawn line (used when the tool is re-armed while it is already on)
    // Returns { getStyle, resetStyle, down, click, arm }; down/click are the button's onmousedown / onclick handlers.
    var _MC_TL_HOLD_MS = 400;
    function _makeTrendlineStyleHold(cfg) {
        var hold  = { timer: null, fired: false, pop: null };
        var style = 'solid';   // 'solid' | 'dotted' — style of the NEXT line drawn. Always 'solid' unless picked from the hold-menu (click / Alt+T reset it).

        function popClose() {
            if (hold.pop && hold.pop.parentNode) hold.pop.parentNode.removeChild(hold.pop);
            hold.pop = null;
        }

        function popOpen(btn) {
            popClose();
            var pop = document.createElement('div');
            pop.className = 'mc-tl-style-pop';
            pop.innerHTML =
                '<div class="mc-tl-style-opt" data-tl-style="solid">' +
                    '<svg width="30" height="8" viewBox="0 0 30 8"><line x1="3" y1="4" x2="27" y2="4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>' +
                    '<span>Solid</span></div>' +
                '<div class="mc-tl-style-opt" data-tl-style="dotted">' +
                    '<svg width="30" height="8" viewBox="0 0 30 8"><line x1="3" y1="4" x2="27" y2="4" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="0.1 5.9"/></svg>' +
                    '<span>Dotted</span></div>';
            var r = btn.getBoundingClientRect();
            pop.style.left = r.left + 'px';
            pop.style.top  = (r.bottom + 6) + 'px';
            document.body.appendChild(pop);
            hold.pop = pop;
        }

        // Option under the pointer (elementFromPoint, not event.target, so it doesn't depend on mouse capture while the button is held)
        function optAt(x, y) {
            if (!hold.pop) return null;
            var el = document.elementFromPoint(x, y);
            var opt = el && el.closest ? el.closest('.mc-tl-style-opt') : null;
            return (opt && hold.pop.contains(opt)) ? opt : null;
        }

        function onMove(evt) {
            if (!hold.pop) return;
            var hot = optAt(evt.clientX, evt.clientY);
            var opts = hold.pop.querySelectorAll('.mc-tl-style-opt');
            for (var i = 0; i < opts.length; i++) opts[i].classList.toggle('hot', opts[i] === hot);
        }

        // Arm the tool with a chosen style. Uses the very same activation path as a click / Alt+T
        // (turns AVWAP / measure off, clears selection and any half-drawn line); if the tool is already armed it just restarts with the new style.
        function arm(s) {
            if (!cfg.isActive()) {
                cfg.toggle();
            } else {
                cfg.cancelDraw();
            }
            style = (s === 'dotted') ? 'dotted' : 'solid';
        }

        function onUp(evt) {
            document.removeEventListener('mouseup',   onUp, true);
            document.removeEventListener('mousemove', onMove, true);
            clearTimeout(hold.timer); hold.timer = null;
            if (!hold.fired) return;   // quick press: the button's own click handler does the normal toggle
            var opt    = optAt(evt.clientX, evt.clientY);
            var picked = opt ? opt.getAttribute('data-tl-style') : null;
            popClose();
            if (picked) arm(picked);
            // Keep `fired` set just long enough to swallow the click that follows this mouseup, then clear it
            setTimeout(function() { hold.fired = false; }, 50);
        }

        return {
            getStyle:   function() { return style; },
            resetStyle: function() { style = 'solid'; },
            arm:        arm,
            // onmousedown on the trendline button
            down: function(evt) {
                if (evt.button !== 0) return;
                var btn = evt.currentTarget;
                clearTimeout(hold.timer);
                hold.fired = false;
                hold.timer = setTimeout(function() {
                    hold.timer = null;
                    hold.fired = true;
                    popOpen(btn);
                }, _MC_TL_HOLD_MS);
                document.addEventListener('mouseup',   onUp, true);
                document.addEventListener('mousemove', onMove, true);
            },
            // onclick on the trendline button — ignored when the press was a hold (the menu handled it)
            click: function() {
                if (hold.fired) return;
                cfg.toggle();
            }
        };
    }

    // Fullscreen chart instance — #mc-fs-trendline-btn (onmousedown="mcFsTlBtnDown(event)" onclick="mcFsTlBtnClick()")
    var _mcFsTlMenu = _makeTrendlineStyleHold({
        toggle:     function() { window.mcFsToggleTrendline(); },
        isActive:   function() { return _mcFsTrendlineMode; },
        cancelDraw: function() {
            _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
            if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
        }
    });
    window.mcFsTlBtnDown    = _mcFsTlMenu.down;
    window.mcFsTlBtnClick   = _mcFsTlMenu.click;
    window.mcFsArmTrendline = _mcFsTlMenu.arm;

    window.mcFsToggleMeasure = function() {
        _mcFsMeasureMode = !_mcFsMeasureMode;
        var btn = document.getElementById('mc-fs-measure-btn');
        if (btn) btn.classList.toggle('active', _mcFsMeasureMode);
        if (_mcFsMeasureMode) {
            // Deactivate trendline and AVWAP when measure is turned on
            if (_mcFsTrendlineMode) {
                _mcFsTrendlineMode = false;
                var tBtn = document.getElementById('mc-fs-trendline-btn');
                if (tBtn) tBtn.classList.remove('active');
                _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
                if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
            }
            if (_mcFsVwapMode) {
                _mcFsVwapMode = false;
                var vBtn = document.getElementById('mc-fs-vwap-btn');
                if (vBtn) vBtn.classList.remove('active');
            }
        } else {
            // Toggled off — clear any live measure
            if (_mcFsMeasureActive || _mcFsMeasurePhase === 1) {
                _mcFsMeasureActive = false;
                _mcFsMeasurePhase  = 0;
                if (_mcFsMeasureRafId) { cancelAnimationFrame(_mcFsMeasureRafId); _mcFsMeasureRafId = null; }
                document.removeEventListener('mousemove', _onMcFsMeasureDragMove);
                document.removeEventListener('mouseup',   _onMcFsMeasureDragEnd);
                document.removeEventListener('mousemove', _onMcFsMeasurePreviewMove);
            }
            _hideMeasureOverlay(_mcFsMeasureSvgOverlay, _mcFsMeasureInfoDiv);
            _mcFsMeasureResult = null;
        }
    };
    window.mcFsToggleTooltip = function() {
        _mcFsTooltipEnabled = !_mcFsTooltipEnabled;
        var btn = document.getElementById('mc-fs-tooltip-btn');
        if (btn) btn.classList.toggle('active', _mcFsTooltipEnabled);
        if (!_mcFsTooltipEnabled && _lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
    };

    // ── Inline symbol switcher ───────────────────────────────────────────────
    window._mcFsSymClick = function() {
        var symEl = document.getElementById('mc-fullscreen-sym');
        if (!symEl || document.getElementById('mc-fs-sym-input')) return;
        var currentSym = symEl.textContent.trim();
        symEl.style.display = 'none';

        // ── Wrapper so dropdown can be positioned relative to input ──────────
        var wrap = document.createElement('span');
        wrap.style.cssText = 'position:relative;display:inline-block;';

        var inp = document.createElement('input');
        inp.id           = 'mc-fs-sym-input';
        inp.type         = 'text';
        inp.value        = '';
        inp.placeholder  = currentSym;
        inp.maxLength    = 10;
        inp.spellcheck   = false;
        inp.autocomplete = 'off';
        inp.style.width  = Math.max(currentSym.length + 2, 6) + 'ch';

        var dd = document.createElement('div');
        dd.id = 'mc-fs-sym-dropdown';
        dd.style.display = 'none';

        wrap.appendChild(inp);
        wrap.appendChild(dd);
        symEl.parentNode.insertBefore(wrap, symEl.nextSibling);
        inp.focus();

        // ── Suggestion engine ─────────────────────────────────────────────────
        var _activeIdx = -1;
        var _results   = [];

        function _getSuggestions(q) {
            if (!q) return [];
            var uq = q.toUpperCase();
            var keys = tickerMap ? Object.keys(tickerMap) : [];
            var prefix = [], substr = [];
            for (var i = 0; i < keys.length; i++) {
                var t = keys[i];
                if (t === uq) { prefix.unshift(t); continue; } // exact first
                if (t.indexOf(uq) === 0) prefix.push(t);
                else if (t.indexOf(uq) > 0) substr.push(t);
            }
            prefix.sort(); substr.sort();
            return prefix.concat(substr).slice(0, 8);
        }

        function _renderDropdown(q) {
            _results = _getSuggestions(q);
            _activeIdx = _results.length ? 0 : -1;
            dd.innerHTML = '';
            if (!q) { dd.style.display = 'none'; return; }
            if (!_results.length) {
                dd.innerHTML = '<div class="mc-fs-dd-empty">No matches in watchlist</div>';
                dd.style.display = '';
                return;
            }
            _results.forEach(function(t, i) {
                var row = tickerMap[t] || {};
                var nameStr = row.name     ? _escHtml(row.name)     : '';
                var indStr  = row.industry ? _escHtml(row.industry)  : (row.sector ? _escHtml(row.sector) : '');
                var el = document.createElement('div');
                el.className = 'mc-fs-dd-row' + (i === 0 ? ' active' : '');
                el.innerHTML =
                    '<span class="mc-fs-dd-ticker">' + _escHtml(t) + '</span>' +
                    (nameStr ? '<span class="mc-fs-dd-name">'  + nameStr + '</span>' : '<span class="mc-fs-dd-name" style="color:var(--border-muted);font-style:italic;">—</span>') +
                    (indStr  ? '<span class="mc-fs-dd-ind">'   + indStr  + '</span>' : '');
                el.addEventListener('mousedown', function(e) {
                    e.preventDefault(); // prevent blur firing before click
                    _selectSym(t);
                });
                dd.appendChild(el);
            });
            dd.style.display = '';
        }

        function _escHtml(s) {
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }

        function _highlightRow(idx) {
            var rows = dd.querySelectorAll('.mc-fs-dd-row');
            rows.forEach(function(r, i) { r.classList.toggle('active', i === idx); });
        }

        function _selectSym(sym) {
            _done = true;
            _restore();
            if (sym && sym !== currentSym) openMcFullscreen(sym, _mcFsTf);
        }

        // ── Lifecycle ─────────────────────────────────────────────────────────
        var _done = false;

        function _confirm() {
            if (_done) return;
            // If a dropdown row is highlighted, use it; else fall back to typed value
            var chosen = (_activeIdx >= 0 && _results[_activeIdx]) ? _results[_activeIdx] : inp.value.trim().toUpperCase();
            _selectSym(chosen || null);
        }

        function _cancel() {
            _done = true;
            _restore();
        }

        function _restore() {
            if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
            symEl.style.display = '';
        }

        inp.addEventListener('input', function() {
            var q = inp.value.trim();
            inp.style.width = Math.max(q.length + 2 || currentSym.length + 2, 6) + 'ch';
            _renderDropdown(q);
        });

        inp.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (_results.length) {
                    _activeIdx = (_activeIdx + 1) % _results.length;
                    _highlightRow(_activeIdx);
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (_results.length) {
                    _activeIdx = (_activeIdx - 1 + _results.length) % _results.length;
                    _highlightRow(_activeIdx);
                }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                _confirm();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                _cancel();
            }
        });

        inp.addEventListener('blur', function() {
            setTimeout(function() { if (!_done) _confirm(); }, 150);
        });

    };

    // ── Watchlist chart symbol click — same behaviour as _mcFsSymClick ────────
    window._wlChartSymClick = function() {
        if (!wlChartTicker) return; // nothing loaded yet
        var symEl = document.getElementById('wl-chart-sym');
        if (!symEl || document.getElementById('wl-chart-sym-input')) return;
        var currentSym = symEl.textContent.trim();
        symEl.style.display = 'none';

        // ── Wrapper so dropdown can be positioned relative to input ──────────
        var wrap = document.createElement('span');
        wrap.style.cssText = 'position:relative;display:inline-block;';

        var inp = document.createElement('input');
        inp.id           = 'wl-chart-sym-input';
        inp.type         = 'text';
        inp.value        = '';
        inp.placeholder  = currentSym;
        inp.maxLength    = 10;
        inp.spellcheck   = false;
        inp.autocomplete = 'off';
        inp.style.width  = Math.max(currentSym.length + 2, 6) + 'ch';

        var dd = document.createElement('div');
        dd.id = 'wl-chart-sym-dropdown';
        dd.style.display = 'none';

        wrap.appendChild(inp);
        wrap.appendChild(dd);
        symEl.parentNode.insertBefore(wrap, symEl.nextSibling);
        inp.focus();

        // ── Suggestion engine ─────────────────────────────────────────────────
        var _activeIdx = -1;
        var _results   = [];

        function _getSuggestions(q) {
            if (!q) return [];
            var uq = q.toUpperCase();
            var keys = tickerMap ? Object.keys(tickerMap) : [];
            var prefix = [], substr = [];
            for (var i = 0; i < keys.length; i++) {
                var t = keys[i];
                if (t === uq) { prefix.unshift(t); continue; }
                if (t.indexOf(uq) === 0) prefix.push(t);
                else if (t.indexOf(uq) > 0) substr.push(t);
            }
            prefix.sort(); substr.sort();
            return prefix.concat(substr).slice(0, 8);
        }

        function _renderDropdown(q) {
            _results = _getSuggestions(q);
            _activeIdx = _results.length ? 0 : -1;
            dd.innerHTML = '';
            if (!q) { dd.style.display = 'none'; return; }
            if (!_results.length) {
                dd.innerHTML = '<div class="mc-fs-dd-empty">No matches in watchlist</div>';
                dd.style.display = '';
                return;
            }
            _results.forEach(function(t, i) {
                var row = tickerMap[t] || {};
                var nameStr = row.name     ? _escHtml(row.name)     : '';
                var indStr  = row.industry ? _escHtml(row.industry)  : (row.sector ? _escHtml(row.sector) : '');
                var el = document.createElement('div');
                el.className = 'mc-fs-dd-row' + (i === 0 ? ' active' : '');
                el.innerHTML =
                    '<span class="mc-fs-dd-ticker">' + _escHtml(t) + '</span>' +
                    (nameStr ? '<span class="mc-fs-dd-name">'  + nameStr + '</span>' : '<span class="mc-fs-dd-name" style="color:var(--border-muted);font-style:italic;">—</span>') +
                    (indStr  ? '<span class="mc-fs-dd-ind">'   + indStr  + '</span>' : '');
                el.addEventListener('mousedown', function(e) {
                    e.preventDefault();
                    _selectSym(t);
                });
                dd.appendChild(el);
            });
            dd.style.display = '';
        }

        function _escHtml(s) {
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }

        function _highlightRow(idx) {
            var rows = dd.querySelectorAll('.mc-fs-dd-row');
            rows.forEach(function(r, i) { r.classList.toggle('active', i === idx); });
        }

        function _selectSym(sym) {
            _done = true;
            _restore();
            if (sym && sym !== currentSym) wlSelectTicker(sym);
        }

        var _done = false;

        function _confirm() {
            if (_done) return;
            var chosen = (_activeIdx >= 0 && _results[_activeIdx]) ? _results[_activeIdx] : inp.value.trim().toUpperCase();
            _selectSym(chosen || null);
        }

        function _cancel() {
            _done = true;
            _restore();
        }

        function _restore() {
            if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
            symEl.style.display = '';
        }

        inp.addEventListener('input', function() {
            var q = inp.value.trim();
            inp.style.width = Math.max(q.length + 2 || currentSym.length + 2, 6) + 'ch';
            _renderDropdown(q);
        });

        inp.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (_results.length) {
                    _activeIdx = (_activeIdx + 1) % _results.length;
                    _highlightRow(_activeIdx);
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (_results.length) {
                    _activeIdx = (_activeIdx - 1 + _results.length) % _results.length;
                    _highlightRow(_activeIdx);
                }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                _confirm();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                _cancel();
            }
        });

        inp.addEventListener('blur', function() {
            setTimeout(function() { if (!_done) _confirm(); }, 150);
        });
    };

    // ── END LW Multichart Infrastructure ─────────────────────────────────────

    function buildMcCellHeader(sym, flagEl) {
        var sd   = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
        var ind  = sd ? (sd.industry || '') : '';
        var pct  = sd ? (sd.Percentile != null ? sd.Percentile : null) : null;
        var wrs  = sd && sd.weighted_rs_pct != null ? Math.round(sd.weighted_rs_pct) : null;
        var live = scanLivePrices && scanLivePrices[sym];
        var dayPct = null;
        if (wlIsMarketOpen() && live && live.price && live.prevClose) {
            dayPct = (live.price - live.prevClose) / live.prevClose * 100;
        } else if (sd && sd.daily != null) {
            dayPct = sd.daily;
        }

        // Industry name + rank in (rank/total) format, percentile-coloured
        var indRankHtml = '';
        if (ind && industriesData && industriesData.industries) {
            var indObj  = industriesData.industries.find(function(x){ return x.industry === ind; });
            var total   = industriesData.industries.length;
            if (indObj && indObj.rank != null) {
                var pctile   = indObj.percentile != null ? indObj.percentile : null;
                var rkColor  = pctile != null ? (pctile >= 75 ? 'var(--success)' : pctile >= 40 ? 'var(--warning-alt)' : 'var(--danger)') : 'var(--text-muted)';
                indRankHtml  = '<span class="mc-cell-hdr-ind-name" style="font-size:0.7em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:30%;margin-left:6px;flex-shrink:1;">' + esc(ind) + '</span>'
                             + '<span class="mc-cell-hdr-rank" style="color:' + rkColor + ';">(' + indObj.rank + '/' + total + ')</span>';
            } else if (ind) {
                indRankHtml  = '<span class="mc-cell-hdr-ind-name" style="font-size:0.7em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:30%;margin-left:6px;flex-shrink:1;">' + esc(ind) + '</span>';
            }
        }

        // RS badges
        var rsBadgeHtml = '';
        if (pct != null) {
            var b = rsBadge(pct);
            if (b) rsBadgeHtml += '<span class="chart-rs-badge ' + b.cls + '" style="font-size:0.62em;padding:1px 5px;">' + b.text + '</span>';
        }
        if (wrs != null) {
            var wCls = wrs >= 75 ? 'rs-high' : wrs >= 40 ? 'rs-mid' : 'rs-low';
            rsBadgeHtml += '<span class="chart-rs-badge ' + wCls + '" style="font-size:0.62em;padding:1px 5px;">' + wrs + '</span>';
        }

        // Price change + chg% combined: "+0.97 (+2.17%)"
        var priceChgHtml = '';
        var chgHtml = '';
        if (dayPct != null) {
            var chgColor = dayPct > 0 ? 'var(--success)' : dayPct < 0 ? 'var(--danger)' : 'var(--border-muted)';
            var chgStyle = 'color:' + chgColor + ';font-size:0.748em;font-weight:600;flex-shrink:0;font-variant-numeric:tabular-nums;white-space:nowrap;';
            var absDelta = null;
            if (wlIsMarketOpen() && live && live.price && live.prevClose) {
                absDelta = live.price - live.prevClose;
            } else if (sd && sd.price != null && dayPct != null) {
                absDelta = sd.price / (1 + dayPct / 100) * (dayPct / 100);
            }
            if (absDelta != null) {
                chgHtml = '<span class="mc-cell-hdr-chg" style="' + chgStyle + '">'
                        + (absDelta >= 0 ? '+' : '') + absDelta.toFixed(2)
                        + ' (' + (dayPct >= 0 ? '+' : '') + dayPct.toFixed(2) + '%)'
                        + '</span>';
            } else {
                chgHtml = '<span class="mc-cell-hdr-chg" style="' + chgStyle + '">'
                        + (dayPct >= 0 ? '+' : '') + dayPct.toFixed(2) + '%'
                        + '</span>';
            }
        }

        var hdr = document.createElement('div');
        hdr.className = 'mc-cell-hdr';
        if (flagEl) hdr.appendChild(flagEl);
        var inner = document.createElement('div');
        inner.className = 'mc-cell-hdr-inner';
        inner.innerHTML = '<span class="mc-cell-hdr-sym">' + esc(sym) + '</span>' + indRankHtml + rsBadgeHtml + '<span style="flex:1;"></span>' + priceChgHtml + chgHtml;
        hdr.appendChild(inner);
        hdr.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            e.stopPropagation();
            var fakeBtn = {
                getAttribute: function(attr) { return attr === 'data-ticker' ? sym : null; },
                getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                _wlNoSwitch: true
            };
            wlOpenPicker(fakeBtn, e, false);
        });
        return hdr;
    }

    // EPS/earnings-date badge -- shown in the fullscreen + watchlist header only,
    // inserted as a sibling right after the 3-month RS badge. Not used in the
    // dense multichart grid cells by design.
    function applyMcEpsBadge(afterEl, fundRow) {
        if (!afterEl) return;
        var existing = document.getElementById('mc-eps-badge');
        var html = '';
        if (fundRow && fundRow.earnings_date) {
            var today = new Date();
            today.setHours(0, 0, 0, 0);
            var ed = new Date(fundRow.earnings_date + 'T00:00:00');
            var days = Math.round((ed - today) / 86400000);
            if (days >= 0 && days <= 30) {
                var urgent = days <= 7;
                var bg = urgent ? 'var(--bg-eps-urgent)' : 'var(--bg-eps-soon)';
                var fg = urgent ? 'var(--text-eps-urgent)' : 'var(--text-eps-soon)';
                var dateLabel = ed.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                html = '<span id="mc-eps-badge" title="' + dateLabel + '" style="background:' + bg + ';color:' + fg
                     + ';font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px;'
                     + 'margin-left:6px;white-space:nowrap;">EPS ' + days + 'd</span>';
            }
        }
        if (existing) {
            if (html) { existing.outerHTML = html; }
            else { existing.parentNode.removeChild(existing); }
        } else if (html) {
            afterEl.insertAdjacentHTML('afterend', html);
        }
    }

    function renderMulticharts() {
        var grid = document.getElementById('multichart-grid');
        if (!grid) return;
        _buildLwMcGrid(grid, mcTickers, mcTimeframe, mcCols, mcWidgets, 'ind');
    }

    window.openMcFullscreen = function(sym, tf, displayName) {
        tf = tf || mcTimeframe || 'D';
        _mcFsTf  = tf;
        _mcFsSym = sym;
        var overlay = document.getElementById('mc-fullscreen-overlay');
        document.getElementById('mc-fullscreen-sym').textContent = displayName || sym;
        var mcFBtn = document.getElementById('mc-fullscreen-details-btn');
        if (mcFBtn) {
            if (displayName) {
                mcFBtn.style.display = 'none';
            } else {
                mcFBtn.style.display = '';
                var mcTicker = sym.replace(/[^A-Z0-9]/gi, '');
                mcFBtn.href = 'https://finviz.com/quote.ashx?t=' + mcTicker;
            }
        }

        // RS badge + industry meta
        var mcPct = null, mcIndustry = '', mcFundRow = null;
        if (snapshot && snapshot.by_industry) {
            outer: for (var ind in snapshot.by_industry) {
                var rows = snapshot.by_industry[ind];
                for (var i = 0; i < rows.length; i++) {
                    if (rows[i].ticker === sym) {
                        mcPct = rows[i].Percentile; mcIndustry = rows[i].industry || ''; mcFundRow = rows[i];
                        break outer;
                    }
                }
            }
        }
        applyRsBadge(document.getElementById('mc-fullscreen-rs-badge'), mcPct, mcFundRow ? mcFundRow.weighted_rs_pct : null, document.getElementById('mc-fullscreen-3mrs-badge'));
        applyMcEpsBadge(document.getElementById('mc-fullscreen-3mrs-badge'), mcFundRow);
        var mcFundStatsEl = document.getElementById('mc-fullscreen-fund-stats');
        if (mcFundStatsEl) mcFundStatsEl.innerHTML = fundStatsHtml(mcFundRow);
        var mcMetaEl = document.getElementById('mc-fullscreen-meta');
        if (mcMetaEl) {
            var mcIndRankHtml = '';
            if (mcIndustry && industriesData && industriesData.industries) {
                var mcIndData = industriesData.industries.find(function(x){ return x.industry === mcIndustry; });
                var mcTotal   = industriesData.industries.length;
                if (mcIndData && mcIndData.rank != null) {
                    var mcPctile  = mcIndData.percentile != null ? mcIndData.percentile : null;
                    var mcRankClr = mcPctile != null ? (mcPctile >= 75 ? 'var(--success)' : mcPctile >= 40 ? 'var(--warning-alt)' : 'var(--danger)') : 'var(--text-muted)';
                    mcIndRankHtml = '<span class="meta-sep">·</span><span class="meta-ind-rank" style="color:' + mcRankClr + '">(' + mcIndData.rank + '/' + mcTotal + ')</span>';
                }
            }
            mcMetaEl.innerHTML = mcIndustry ? industryLinkHtml(mcIndustry, 'closeMcFullscreen') + mcIndRankHtml : '';
            mcMetaEl.style.display = mcIndustry ? '' : 'none';
        }

        // Reset VWAP state for new symbol
        _mcFsVwapMode = false; _mcFsVwapSeries = []; _mcFsSelectedVwapIdx = -1;
        var vwapBtn  = document.getElementById('mc-fs-vwap-btn');
        if (vwapBtn)  vwapBtn.classList.remove('active');
        // Drop any half-finished measurement so its anchor can't leak onto the new symbol
        _mcFsResetMeasure();
        // Reset trendlines for new symbol (keep mode active if it was on)
        _mcFsTrendlines = []; _mcFsTrendlineFirst = null;
        if (_mcFsTrendSvgOverlay) _mcFsTrendSvgOverlay.style.display = 'none';
        _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;

        // Sync TF buttons + default viewport
        document.querySelectorAll('.mc-fs-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        _mcFsVisibleBars = tf === 'D' ? 252 : tf === 'W' ? 104 : 60;

        // Sync MA badge with current state
        _mcFsUpdateMaBadge();
        // Ensure MA panel is closed when opening a new symbol
        var maPanel   = document.getElementById('mc-fs-ma-panel');
        var maChevron = document.getElementById('mc-fs-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';

        overlay.classList.add('open');
        updateQueueButtons();

        // Show loading state then fetch + render LW chart
        var container = document.getElementById('mc-fullscreen-chart');
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;">Loading…</div>';
        var openSym = sym;
        // No forced cache-clear here anymore — see the matching note in
        // mcFsSetTf. Reopening a symbol/TF already fetched this session now
        // renders instantly from cache instead of re-entering the queue.
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (!document.getElementById('mc-fullscreen-overlay').classList.contains('open')) return;
            if (_mcFsSym !== openSym || _mcFsTf !== tf) return; // a newer symbol or timeframe has since superseded this fetch
            if (_mcFsChart && _mcFsBuiltSym === sym && _mcFsBuiltTf === tf) return; // truly already rendered for this exact symbol/tf
            _buildFsChart(sym, ohlcv, tf);
        });
    };

    window.closeMcFullscreen = function() {
        document.getElementById('mc-fullscreen-overlay').classList.remove('open');
        _mcFsResetMeasure();
        _mcFsDismissCtx();
        _mcFsStopLiveTick();
        if (_mcFsChart) { try { _mcFsChart.remove(); } catch(e) {} _mcFsChart = null; }
        _mcFsBuiltSym = null; _mcFsBuiltTf = null;
        _mcFsCandle = null; _mcFsVol = null; _mcFsVolMa = null; _mcFsMaSeries = {}; _mcFsVwapSeries = [];
        _mcFsTrendlines = []; _mcFsTrendlineFirst = null;
        _mcFsTrendSvgOverlay = null; _mcFsTrendSvgLine = null; // removed with chart container
        _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
        _mcFsTrendlineMode = false; _mcFsSelectedTrendlineIdx = -1; _mcFsSelectedVwapIdx = -1;
        var tBtn  = document.getElementById('mc-fs-trendline-btn');
        if (tBtn)  tBtn.classList.remove('active');
        _mcFsMaDataMap = {}; _mcFsLastCrosshairTime = null;
        _mcFsVolSmaMap = null;
        if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
        _mcFsSym = null;
        var _mktEl = document.getElementById('mc-fs-mkt-info');
        if (_mktEl) _mktEl.style.display = 'none';
        if (_mcFsKeyHandler) { document.removeEventListener('keydown', _mcFsKeyHandler); _mcFsKeyHandler = null; }
        document.getElementById('mc-fullscreen-chart').innerHTML = '';
        // Close MA panel if open
        var maPanel   = document.getElementById('mc-fs-ma-panel');
        var maChevron = document.getElementById('mc-fs-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
        // Always clear the scan-nav-panel state so stale preset data never persists
        if (typeof snpHide === 'function') snpHide();
        // Grid queues gated in _drainMcQueue while this overlay was open can
        // resume now rather than waiting out their next 1s poll timer.
        Object.keys(_mcFetchQueue).forEach(function(qKey) {
            if (qKey.slice(-5) === '_grid') _drainMcQueue(qKey.slice(0, -5), true);
        });
    };

    // ══════════════════════════════════════════════════════════════════════
    // WATCHLIST LW CHART — full parallel to the mc-fullscreen chart system
    // ══════════════════════════════════════════════════════════════════════

    // ── Trendline primitive ───────────────────────────────────────────────
    function _addWlTrendline(p1, p2, extend, dotted) {
        return _addTrendlineCore(p1, p2, _wlChart, _wlCandle, _wlOhlcv, _wlTrendlines, { extend: !!extend, dotted: !!dotted });
    }

    // Watchlist-chart equivalent of _addFsVwap (the click handler builds its AVWAP inline, so restore needs this).
    function _addWlVwap(anchorIdx) {
        if (!_wlChart || !_wlOhlcv.length) return;
        var data = _calcAVWAP(_wlOhlcv, anchorIdx);
        if (!data.length) return;
        var s = _wlChart.addSeries(LightweightCharts.LineSeries, { color: _AVWAP_COLOR, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
        s.setData(data);
        _wlVwapSeries.push({ series: s, anchor: anchorIdx, color: _AVWAP_COLOR, dataMap: new Map(data.map(function(d) { return [d.time, d.value]; })) });
    }

    // ── Hit-test helpers ─────────────────────────────────────────────────
    function _wlTrendlineHitTest(clientX, clientY) {
        return _trendlineHitTestCore(clientX, clientY, _wlChart, _wlCandle, _wlOhlcv, _wlTrendlines, _wlTrendContRef);
    }

    function _wlDeselectAllTrendlines() {
        _deselectAllTrendlinesCore(_wlTrendlines);
        _wlSelectedTrendlineIdx = -1;
    }

    function _wlSelectVwap(idx) {
        _selectVwapCore(_wlVwapSeries, idx);
        _wlSelectedVwapIdx = idx;
    }
    function _wlDeselectAllVwaps() {
        _deselectAllVwapsCore(_wlVwapSeries);
        _wlSelectedVwapIdx = -1;
    }

    function _wlVwapHitTest(clientX, clientY) {
        return _vwapHitTestCore(clientX, clientY, _wlChart, _wlVwapSeries, _wlLastCrosshairTime, 'wl-chart-widget');
    }

    function _wlAnchorHitTest(clientX, clientY, tlIdx) {
        return _anchorHitTestCore(clientX, clientY, tlIdx, _wlTrendlines, _wlChart, _wlCandle, _wlOhlcv, _wlTrendContRef);
    }

    // ── Anchor drag ───────────────────────────────────────────────────────
    function _onWlTrendAnchorDragMove(evt) {
        _onTrendAnchorDragMoveCore(evt, {
            dragState:  _wlTrendDragState,
            chart:      _wlChart,
            candle:     _wlCandle,
            contRef:    _wlTrendContRef,
            trendlines: _wlTrendlines,
            ohlcv:      _wlOhlcv,
            svgOverlay: _wlTrendSvgOverlay,
            svgLine:    _wlTrendSvgLine
        });
    }

    function _onWlTrendAnchorDragEnd() {
        _onTrendAnchorDragEndCore({
            getDragState: function() { return _wlTrendDragState; },
            setDragState: function(v) { _wlTrendDragState = v; },
            getSym:       function() { return _wlSym; },
            trendlines:   _wlTrendlines,
            contRef:      _wlTrendContRef,
            svgOverlay:   _wlTrendSvgOverlay,
            moveHandler:  _onWlTrendAnchorDragMove,
            endHandler:   _onWlTrendAnchorDragEnd
        });
    }

    // Abandon a half-finished watchlist measurement and drop the committed ones (see _mcFsResetMeasure).
    function _wlResetMeasure() {
        _wlMeasureActive = false; _wlMeasurePhase = 0; _wlMeasureStart = null; _wlMeasureResult = null;
        _measureClearAll(_wlMeasureList);
        if (_wlMeasureRafId) { cancelAnimationFrame(_wlMeasureRafId); _wlMeasureRafId = null; }
        document.removeEventListener('mousemove', _onWlMeasurePreviewMove);
        _hideMeasureOverlay(_wlMeasureSvgOverlay, _wlMeasureInfoDiv);
    }

    // ── WL Measure drag handlers ─────────────────────────────────────────────
    function _onWlMeasureDragMove(evt) {
        _onMeasureDragMoveCore(evt, {
            getActive:  function() { return _wlMeasureActive; },
            contRef:    _wlTrendContRef,
            chart:      _wlChart,
            candle:     _wlCandle,
            ohlcv:      _wlOhlcv,
            getStart:   function() { return _wlMeasureStart; },
            getRafId:   function() { return _wlMeasureRafId; },
            setRafId:   function(v) { _wlMeasureRafId = v; },
            setResult:  function(v) { _wlMeasureResult = v; },
            svgOverlay: _wlMeasureSvgOverlay,
            svgRect:    _wlMeasureSvgRect,
            hLine:      _wlMeasureHLine,
            infoDiv:    _wlMeasureInfoDiv
        });
    }
    function _onWlMeasureDragEnd() {
        _onMeasureDragEndCore({
            moveHandler: _onWlMeasureDragMove,
            endHandler:  _onWlMeasureDragEnd,
            setActive:   function(v) { _wlMeasureActive = v; }
        });
    }
    function _onWlMeasurePreviewMove(evt) {
        _onMeasurePreviewMoveCore(evt, {
            getActive:  function() { return _wlMeasureActive; },
            getPhase:   function() { return _wlMeasurePhase; },
            contRef:    _wlTrendContRef,
            chart:      _wlChart,
            candle:     _wlCandle,
            ohlcv:      _wlOhlcv,
            getStart:   function() { return _wlMeasureStart; },
            getRafId:   function() { return _wlMeasureRafId; },
            setRafId:   function(v) { _wlMeasureRafId = v; },
            setResult:  function(v) { _wlMeasureResult = v; },
            svgOverlay: _wlMeasureSvgOverlay,
            svgRect:    _wlMeasureSvgRect,
            hLine:      _wlMeasureHLine,
            infoDiv:    _wlMeasureInfoDiv
        });
    }

    // ── Trendline mousedown (capture phase, blocks LW canvas pan) ─────────
    function _onWlTrendMouseDown(evt) {
        _onTrendMouseDownCore(evt, {
            candle:  _wlCandle,
            chart:   _wlChart,
            contRef: _wlTrendContRef,
            getMeasureMode:    function() { return _wlMeasureMode; },
            getDragState:      function() { return _wlTrendDragState; },
            setDragState:      function(v) { _wlTrendDragState = v; },
            trendDraw:         _wlTrendDraw,
            svgOverlay:        _wlTrendSvgOverlay,
            svgLine:           _wlTrendSvgLine,
            ohlcv:             _wlOhlcv,
            getMeasurePhase:   function() { return _wlMeasurePhase; },
            setMeasurePhase:   function(v) { _wlMeasurePhase = v; },
            getMeasureResult:  function() { return _wlMeasureResult; },
            setMeasureResult:  function(v) { _wlMeasureResult = v; },
            setMeasureActive:  function(v) { _wlMeasureActive = v; },
            getMeasureRafId:   function() { return _wlMeasureRafId; },
            setMeasureRafId:   function(v) { _wlMeasureRafId = v; },
            getMeasureStart:   function() { return _wlMeasureStart; },
            setMeasureStart:   function(v) { _wlMeasureStart = v; },
            measureSvgOverlay: _wlMeasureSvgOverlay,
            measureSvgRect:    _wlMeasureSvgRect,
            measureHLine:      _wlMeasureHLine,
            measureInfoDiv:    _wlMeasureInfoDiv,
            measureList:       _wlMeasureList,
            measurePreviewMoveHandler: _onWlMeasurePreviewMove,
            getSelectedIdx:    function() { return _wlSelectedTrendlineIdx; },
            setSelectedIdx:    function(v) { _wlSelectedTrendlineIdx = v; },
            trendlines:        _wlTrendlines,
            deselectAllTrendlines: _wlDeselectAllTrendlines,
            deselectAllVwaps:  _wlDeselectAllVwaps,
            anchorHitTest:     _wlAnchorHitTest,
            trendlineHitTest:  _wlTrendlineHitTest,
            dragMoveHandler:   _onWlTrendAnchorDragMove,
            dragEndHandler:    _onWlTrendAnchorDragEnd,
            getTrendlineMode:  function() { return _wlTrendlineMode; },
            setTrendlineMode:  function(v) { _wlTrendlineMode = v; },
            getTrendlineStyle: function() { return _wlTlMenu.getStyle(); },
            getLastCrosshairTime: function() { return _wlLastCrosshairTime; },
            addTrendline:      _addWlTrendline,
            onTrendDrawn:      function(tl) { _cdAddTl(_wlSym, tl); },
            onMeasureCommitted: function(ms) { ms.sym = _wlSym; _cdAddMs(ms.sym, ms); },
            doneBtnId:         'wl-chart-trendline-btn'
        });
    }

    // ── Trendline SVG mousemove preview ───────────────────────────────────
    function _onWlTrendMouseMove(evt) {
        _onTrendMouseMoveCore(evt, {
            getTrendDraw:    function() { return _wlTrendDraw; },
            svgOverlay:      _wlTrendSvgOverlay,
            svgLine:         _wlTrendSvgLine,
            candle:          _wlCandle,
            chart:           _wlChart,
            contRef:         _wlTrendContRef,
            trendlines:      _wlTrendlines,
            getTrendlineMode: function() { return _wlTrendlineMode; },
            getDragState:    function() { return _wlTrendDragState; },
            getSelectedIdx:  function() { return _wlSelectedTrendlineIdx; },
            anchorHitTest:   _wlAnchorHitTest,
            trendlineHitTest: _wlTrendlineHitTest,
            ohlcv:           _wlOhlcv,
            getMeasureMode:  function() { return _wlMeasureMode; },
            getMeasurePhase: function() { return _wlMeasurePhase; },
            measureList:     _wlMeasureList
        });
    }

    // ── Right-click context menu ──────────────────────────────────────────
    function _wlDismissCtx() {
        _hideCtxMenu('wl-chart-ctx-menu');
        _wlCtxPrice     = null;
        _wlCtxMa        = null;
        _wlCtxTrendline = null;
        _wlCtxAvwap     = null;
    }

    window.wlCtxAlert = function(direction) {
        _ctxAlertCore(direction, {
            getCtxTrendline: function() { return _wlCtxTrendline; },
            getCtxAvwap:     function() { return _wlCtxAvwap; },
            getCtxMa:        function() { return _wlCtxMa; },
            getCtxPrice:     function() { return _wlCtxPrice; },
            getSym:          function() { return _wlSym; },
            getTf:           function() { return _wlTf; },
            dismiss:         _wlDismissCtx
        });
    };

    function _wlAttachCtxMenu() {
        _attachCtxMenuCore({
            getAttached: function() { return _wlCtxAttached; },
            setAttached: function(v) { _wlCtxAttached = v; },
            parentElId: 'wl-chart-body',
            chartDivId: 'wl-chart-widget',
            getTooltipEnabled: function() { return _wlTooltipEnabled; },
            setTooltipEnabled: function(v) { _wlTooltipEnabled = v; },
            tooltipBtnId: 'wl-chart-tooltip-btn',
            getMeasurePhase:  function() { return _wlMeasurePhase; },
            setMeasurePhase:  function(v) { _wlMeasurePhase = v; },
            setMeasureActive: function(v) { _wlMeasureActive = v; },
            getMeasureRafId:  function() { return _wlMeasureRafId; },
            setMeasureRafId:  function(v) { _wlMeasureRafId = v; },
            measurePreviewMoveHandler: _onWlMeasurePreviewMove,
            getMeasureSvgOverlay: function() { return _wlMeasureSvgOverlay; },
            getMeasureInfoDiv:    function() { return _wlMeasureInfoDiv; },
            getMeasureResult: function() { return _wlMeasureResult; },
            setMeasureResult: function(v) { _wlMeasureResult = v; },
            trendDraw:     _wlTrendDraw,
            getSvgOverlay: function() { return _wlTrendSvgOverlay; },
            getVwapMode: function() { return _wlVwapMode; },
            setVwapMode: function(v) { _wlVwapMode = v; },
            vwapBtnId: 'wl-chart-vwap-btn',
            getChart: function() { return _wlChart; },
            getSym:   function() { return _wlSym; },
            trendlineHitTest: _wlTrendlineHitTest,
            getTrendlines: function() { return _wlTrendlines; },
            ctxAboveTxtId:  'wl-chart-ctx-above-txt',
            ctxBelowTxtId:  'wl-chart-ctx-below-txt',
            ctxMenuId:      'wl-chart-ctx-menu',
            setCtxTrendline: function(v) { _wlCtxTrendline = v; },
            setCtxPrice:     function(v) { _wlCtxPrice = v; },
            setCtxMa:        function(v) { _wlCtxMa = v; },
            vwapHitTest: _wlVwapHitTest,
            getVwapSeries: function() { return _wlVwapSeries; },
            getOhlcv:      function() { return _wlOhlcv; },
            setCtxAvwap: function(v) { _wlCtxAvwap = v; },
            getCandle: function() { return _wlCandle; },
            getLastCrosshairPrice: function() { return _wlLastCrosshairPrice; },
            getLastCrosshairTime:  function() { return _wlLastCrosshairTime; },
            getMaDataMap: function() { return _wlMaDataMap; },
            getMaSeries:  function() { return _wlMaSeries; },
            dismissCtx: _wlDismissCtx
        });
    }

    // ── Core chart builder ────────────────────────────────────────────────
    function _destroyWlChart() {
        _wlResetMeasure();
        if (_wlChart) { try { _wlChart.remove(); } catch(e) {} _wlChart = null; }
        _wlCandle = null; _wlVol = null; _wlVolMa = null; _wlVolData = null; _wlMaSeries = {}; _wlVwapSeries = [];
        _wlTrendlines = []; _wlTrendlineFirst = null;
        _wlTrendSvgOverlay = null; _wlTrendSvgLine = null;
        _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
        _wlTrendlineMode = false; _wlSelectedTrendlineIdx = -1; _wlSelectedVwapIdx = -1;
        _wlMaDataMap = {}; _wlLastCrosshairTime = null; _wlSym = null;
        _wlVolSmaMap = null;
        if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
        if (_wlKeyHandler) { document.removeEventListener('keydown', _wlKeyHandler); _wlKeyHandler = null; }
        var mktEl = document.getElementById('wl-chart-mkt-info');
        if (mktEl) mktEl.style.display = 'none';
        var tBtn  = document.getElementById('wl-chart-trendline-btn');
        if (tBtn)  tBtn.classList.remove('active');
        var vBtn  = document.getElementById('wl-chart-vwap-btn');
        if (vBtn)  vBtn.classList.remove('active');
        var ttBtn = document.getElementById('wl-chart-tooltip-btn');
        if (ttBtn) ttBtn.classList.toggle('active', _wlTooltipEnabled);
        var maPanel   = document.getElementById('wl-chart-ma-panel');
        var maChevron = document.getElementById('wl-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
    }

    function _buildWlChart(sym, ohlcv, tf) {
        var container = document.getElementById('wl-chart-widget');
        container.innerHTML = '';
        _destroyWlChart();

        _wlOhlcv = ohlcv || [];
        _wlSym   = sym;
        _wlTf    = tf;
        _wlLastCrosshairPrice = null;

        if (!window.LightweightCharts || !_wlOhlcv.length) {
            var _wlMsg = ohlcv === null ? 'Failed to load — click to retry' : 'No data';
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;' + (ohlcv === null ? 'cursor:pointer;text-decoration:underline;' : '') + '">' + _wlMsg + '</div>';
            if (ohlcv === null) {
                container.querySelector('div').addEventListener('click', function() {
                    fetchMcOhlcv(sym, tf).then(function(retryOhlcv) { _buildWlChart(sym, retryOhlcv, tf); });
                });
            }
            return;
        }

        // SVG trendline overlay
        _wlTrendContRef = container;
        container.removeEventListener('mousedown', _onWlTrendMouseDown, true);
        container.addEventListener('mousedown', _onWlTrendMouseDown, true);

        var _existingSvg = container.querySelector('.wl-trend-svg-overlay');
        if (_existingSvg) {
            _wlTrendSvgOverlay = _existingSvg;
            _wlTrendSvgLine    = _existingSvg.querySelector('line');
        } else {
            _wlTrendSvgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            _wlTrendSvgOverlay.setAttribute('class', 'wl-trend-svg-overlay');
            _wlTrendSvgOverlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5;display:none;';
            _wlTrendSvgLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            _wlTrendSvgLine.setAttribute('stroke', _TRENDLINE_COLOR());
            _wlTrendSvgLine.setAttribute('stroke-width', '1.5');
            _wlTrendSvgLine.setAttribute('x1', '0'); _wlTrendSvgLine.setAttribute('y1', '0');
            _wlTrendSvgLine.setAttribute('x2', '0'); _wlTrendSvgLine.setAttribute('y2', '0');
            _wlTrendSvgOverlay.appendChild(_wlTrendSvgLine);
            container.style.position = 'relative';
            container.appendChild(_wlTrendSvgOverlay);
        }
        _wlTrendSvgOverlay.style.display = 'none';

        // ── Measure tool overlay ───────────────────────────────────────────
        var _wlmOver = _ensureMeasureOverlay(container, 'wl-measure-svg', 'wl-measure-info');
        _wlMeasureSvgOverlay = _wlmOver.svg;
        _wlMeasureSvgRect    = _wlmOver.rect;
        _wlMeasureHLine      = _wlmOver.hLine;
        _wlMeasureInfoDiv    = _wlmOver.info;
        _wlMeasureResult     = null;
        _hideMeasureOverlay(_wlMeasureSvgOverlay, _wlMeasureInfoDiv);

        container.removeEventListener('mousemove', _onWlTrendMouseMove);
        container.addEventListener('mousemove', _onWlTrendMouseMove);

        // Create LW chart — identical options to fullscreen
        _wlChart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted'), panes: { separatorColor: themeColor('bg-subtle'), separatorHoverColor: themeColor('bg-surface-alpha') } },
            grid:    { vertLines: { visible: false }, horzLines: { visible: false } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted'), scaleMargins: { top: 0.05, bottom: 0.02 } },
            timeScale: { borderColor: themeColor('bg-surface'), timeVisible: false, secondsVisible: false, rightOffset: 24 },
            localization: { timeFormatter: _mcLwCrosshairDateFmt },
            handleScroll: true, handleScale: true,
        });
        _wlAttachCtxMenu();

        // Watermark — same as fullscreen: ticker + company name, bottom-right
        // (top corners here are also taken, by the legend and pre/post badge
        // just below). Meta comes from the same cache fetchMcOhlcv already
        // populates, no extra request.
        var _wlMeta        = _mcMetaCache[sym] || {};
        var _wlCompanyName = _wlMeta.longName || _wlMeta.shortName || '';
        _wlWatermark = LightweightCharts.createTextWatermark(_wlChart.panes()[0], {
            horzAlign: 'right',
            vertAlign: 'bottom',
            lines: [
                { text: sym, color: themeColor('chart-watermark'), fontSize: _MC_WM_SYM_SIZE, fontStyle: _MC_WM_SYM_STYLE },
                _wlCompanyName ? { text: _wlCompanyName, color: themeColor('chart-watermark'), fontSize: _MC_WM_NAME_SIZE } : null,
            ].filter(Boolean),
        });

        // Candle series
        _wlCandle = _wlChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'), borderVisible: false,
            wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
            priceLineVisible: false, lastValueVisible: true,
        });
        _wlCandle.setData(_wlOhlcv);

        // Volume pane
        _wlVol = _wlChart.addSeries(LightweightCharts.HistogramSeries, {
            color: themeColor('al-chart-volume'), priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: true,
        }, 1);
        _wlVol.setData(_wlOhlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
        }));
        _wlVol.priceScale().applyOptions({ visible: true, borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted'), minimumWidth: 60 });

        // 50 SMA on volume
        (function() {
            var period = 50;
            _wlVolData = [];
            for (var i = period - 1; i < _wlOhlcv.length; i++) {
                var sum = 0;
                for (var j = i - (period - 1); j <= i; j++) sum += (_wlOhlcv[j].volume || 0);
                _wlVolData.push({ time: _wlOhlcv[i].time, value: sum / period });
            }
            _wlVolMa = _wlChart.addSeries(LightweightCharts.LineSeries, {
                color: '#1848cc', lineWidth: 1,
                priceLineVisible: false, lastValueVisible: true,
                crosshairMarkerVisible: false,
            }, 1);
            _wlVolMa.setData(_wlVolData);
        })();
        _wlVolSmaMap = _wlVolData && _wlVolData.length
            ? new Map(_wlVolData.map(function(d) { return [d.time, d.value]; }))
            : null;

        // Pin volume pane to ~22% height
        (function() {
            var panes = _wlChart.panes();
            if (panes && panes.length >= 2) {
                var totalH = container ? container.offsetHeight : 700;
                panes[1].setHeight(Math.round(totalH * 0.22));
            }
        })();

        // Vol % vs 50-SMA label
        (function() {
            if (!_wlVolData || !_wlVolData.length || !_wlOhlcv.length) return;
            var lastBar = _wlOhlcv[_wlOhlcv.length - 1];
            var lastVol = lastBar.volume;
            var sma50   = _wlVolData[_wlVolData.length - 1].value;
            if (!sma50) return;
            function nthSunday(yr, mo, n) {
                var d = new Date(Date.UTC(yr, mo, 1));
                return new Date(Date.UTC(yr, mo, 1 + (7 - d.getUTCDay()) % 7 + (n - 1) * 7));
            }
            var now     = Date.now();
            var barDate = new Date(lastBar.time * 1000);
            var yr = barDate.getUTCFullYear(), mo = barDate.getUTCMonth(), dy = barDate.getUTCDate();
            var isDST   = barDate >= nthSunday(yr, 2, 2) && barDate < nthSunday(yr, 10, 1);
            var etDelta = isDST ? 4 : 5;
            var mktOpen  = new Date(Date.UTC(yr, mo, dy,  9 + etDelta, 30));
            var mktClose = new Date(Date.UTC(yr, mo, dy, 16 + etDelta,  0));
            var totalMs  = mktClose - mktOpen;
            var timeratio = 1.0;
            if (now > mktOpen && now < mktClose) timeratio = totalMs / (now - mktOpen);
            var projectedVol = lastVol * timeratio;
            var volDiffPct   = (projectedVol / sma50 - 1) * 100;
            var sign  = volDiffPct >= 0 ? '+' : '';
            var color = volDiffPct >= 0 ? 'var(--success)' : 'var(--danger)';
            var lbl = document.createElement('div');
            lbl.id = 'wl-chart-vol-pct-label';
            lbl.style.cssText = 'position:absolute;z-index:20;pointer-events:none;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:3px;white-space:nowrap;line-height:1;';
            lbl.innerHTML = '<span style="color:var(--border-muted);">›</span>'
                          + '<span style="color:' + color + ';">' + sign + volDiffPct.toFixed(1) + '%</span>';
            container.appendChild(lbl);
            setTimeout(function() {
                if (!_wlChart) return;
                var volPaneTop = 0, volPaneH = 0;
                try {
                    var panes = _wlChart.panes();
                    var pe = (panes && panes[1] && typeof panes[1].getElement === 'function') ? panes[1].getElement() : null;
                    if (pe) {
                        var r = pe.getBoundingClientRect();
                        var cr = container.getBoundingClientRect();
                        volPaneTop = r.top  - cr.top;
                        volPaneH   = r.height;
                    }
                } catch(e) {}
                if (!volPaneH) {
                    var totalH = container.offsetHeight;
                    volPaneH   = Math.round(totalH * 0.22);
                    volPaneTop = totalH - volPaneH - 22;
                }
                var lblTop = (volPaneTop + volPaneH - 28) + 'px';
                function positionVolLabel() {
                    if (!lbl.isConnected || !_wlChart) return;
                    var lastX = _wlChart.timeScale().timeToCoordinate(lastBar.time);
                    // Plot area = container minus the right price scale. Anything past
                    // that bleeds into the watchlist panel (DOM div, not canvas).
                    var scaleW = 0;
                    try { scaleW = _wlChart.priceScale('right').width(); } catch (e) {}
                    var plotW = container.clientWidth - scaleW;
                    lbl.style.display = 'flex';          // must be visible to measure
                    if (lastX == null || lastX < 0 || lastX + 10 + lbl.offsetWidth > plotW) {
                        lbl.style.display = 'none'; return;
                    }
                    lbl.style.left = (lastX + 10) + 'px';
                    lbl.style.top  = lblTop;
                }
                positionVolLabel();
                // Logical range fires on every pan/zoom step (time-range lagged by up to a bar).
                _wlChart.timeScale().subscribeVisibleLogicalRangeChange(positionVolLabel);
            }, 60);
        })();

        // Active MAs
        Object.keys(_wlActiveMas).forEach(function(key) {
            if (!_wlActiveMas[key]) return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = _wlChart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
            var maData = _calcMA(_wlOhlcv, key);
            s.setData(maData);
            _wlMaSeries[key]  = s;
            _wlMaDataMap[key] = new Map(maData.map(function(d) { return [d.time, d.value]; }));
        });

        // Visible range
        var n = _wlOhlcv.length;
        _wlChart.timeScale().setVisibleLogicalRange({ from: n - _wlVisibleBars, to: n + 15 });

        // Re-render measure overlay on pan/zoom
        _wlChart.timeScale().subscribeVisibleLogicalRangeChange(function() {
            if (_wlMeasureResult) {
                _renderMeasureOverlay(_wlChart, _wlCandle, _wlTrendContRef,
                    _wlMeasureSvgOverlay, _wlMeasureSvgRect, _wlMeasureHLine,
                    _wlMeasureInfoDiv, _wlMeasureResult);
            }
            _measureRenderAll(_wlMeasureList, _wlChart, _wlCandle, _wlTrendContRef, _wlOhlcv);
        });

        // Click: AVWAP + selection
        _wlChart.subscribeClick(function(param) {
            if (_wlVwapMode) {
                if (!param.time) return;
                var idx = _barIdxByTime(_wlOhlcv, param.time);
                if (idx < 0) return;
                var color = _AVWAP_COLOR;
                var data  = _calcAVWAP(_wlOhlcv, idx);
                var dataMap = new Map(data.map(function(d) { return [d.time, d.value]; }));
                var s = _wlChart.addSeries(LightweightCharts.LineSeries, { color: color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
                s.setData(data);
                _wlVwapSeries.push({ series: s, anchor: idx, color: color, dataMap: dataMap });
                _cdAddAv(_wlSym, _wlOhlcv, _wlTf, idx);   // save the anchor
                // Select the new AVWAP so Delete removes it straight away (trendlines deselected first, as on fullscreen).
                _wlDeselectAllTrendlines();
                _wlSelectVwap(_wlVwapSeries.length - 1);
                return;
            }
            if (_wlTrendlineMode) return;
            if (!_wlVwapSeries.length || !param.time || !param.point) {
                if (_wlSelectedVwapIdx !== -1) _wlDeselectAllVwaps();
                return;
            }
            var HIT_PX = 8;
            var hitIdx = -1;
            _wlVwapSeries.forEach(function(entry, i) {
                if (!entry.dataMap) return;
                var avwapVal = entry.dataMap.get(param.time);
                if (avwapVal == null) return;
                var yCoord = entry.series.priceToCoordinate(avwapVal);
                if (yCoord == null) return;
                if (Math.abs(param.point.y - yCoord) <= HIT_PX) hitIdx = i;
            });
            if (hitIdx !== -1) {
                if (_wlSelectedVwapIdx === hitIdx) { _wlDeselectAllVwaps(); }
                else { _wlSelectVwap(hitIdx); }
            } else {
                if (_wlSelectedVwapIdx !== -1) _wlDeselectAllVwaps();
            }
        });

        // OHLC legend
        var leg = document.createElement('div');
        leg.id = 'wl-chart-legend';
        leg.style.cssText = 'position:absolute;top:8px;left:14px;z-index:10;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-muted-2);pointer-events:none;line-height:1.8;background:var(--bg-page-alpha-3);padding:4px 10px;border-radius:4px;';
        container.style.position = 'relative';
        container.appendChild(leg);

        // Pre/post-market price badge — same treatment as the fullscreen chart:
        // no background/border, right-offset computed dynamically off the
        // price-scale's actual rendered width so it never overlaps axis labels.
        var wlPrepost = document.createElement('div');
        wlPrepost.id = 'wl-chart-prepost-badge';
        wlPrepost.style.cssText = 'position:absolute;top:8px;right:8px;z-index:10;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;pointer-events:none;line-height:1.6;padding:3px 8px;border-radius:4px;display:none;';
        container.appendChild(wlPrepost);
        _wlRenderPrePostBadge(sym);

        function fp(v) { return v != null ? v.toFixed(2) : '—'; }
        function fv(v) { return v==null?'—':v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':v.toFixed(0); }

        _wlChart.subscribeCrosshairMove(function(p) {
            if (p.point && _wlCandle) {
                var cursorPrice = _wlCandle.coordinateToPrice(p.point.y);
                _wlLastCrosshairPrice = (cursorPrice != null && !isNaN(cursorPrice)) ? cursorPrice : null;
            } else {
                _wlLastCrosshairPrice = null;
            }
            _wlLastCrosshairTime = p.time || null;
            if (!p.time || !p.seriesData || !p.seriesData.size) {
                leg.innerHTML = '';
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
                return;
            }
            var d = p.seriesData.get(_wlCandle);
            if (!d) {
                leg.innerHTML = '';
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
                return;
            }
            var cl = d.close >= d.open ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
            var vd = p.seriesData.get(_wlVol);
            var chgHtml = '';
            var barIdx = _barIdxByTime(_wlOhlcv, p.time);
            if (barIdx > 0) {
                var prevClose = _wlOhlcv[barIdx - 1].close;
                var delta = d.close - prevClose;
                var pct = (delta / prevClose) * 100;
                var chgClr = delta >= 0 ? 'var(--success)' : 'var(--danger)';
                chgHtml = '&nbsp;&nbsp;<span style="color:' + chgClr + '">'
                        + (delta >= 0 ? '+' : '') + delta.toFixed(2)
                        + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)'
                        + '</span>';
            }
            leg.innerHTML =
                '<span style="color:var(--text-muted-2)">O</span> <span style="color:'+cl+'">'+fp(d.open)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">H</span> <span style="color:'+cl+'">'+fp(d.high)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">L</span> <span style="color:'+cl+'">'+fp(d.low)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">C</span> <span style="color:'+cl+'">'+fp(d.close)+'</span>' +
                chgHtml +
                (vd ? '&nbsp;&nbsp;<span style="color:var(--text-muted)">Vol</span> <span style="color:var(--text-muted-2)">' + fv(vd.value) + '</span>' : '');
            // Floating tooltip
            if (_wlTooltipEnabled) {
                var ttDiv = _getLwTooltipDiv();
                ttDiv.innerHTML = _buildTooltipHtml(d, barIdx, _wlOhlcv, _wlVolSmaMap, _wlMaDataMap, _wlActiveMas, p.time);
                ttDiv.style.display = 'block';
                if (p.point) {
                    var rect = container.getBoundingClientRect();
                    _positionTooltip(ttDiv, rect.left + p.point.x, rect.top + p.point.y, rect.right);
                }
            } else if (_lwTooltipDiv) {
                _lwTooltipDiv.style.display = 'none';
            }
        });

        // Market info bar
        (function() {
            if (!_wlOhlcv.length) return;
            var last    = _wlOhlcv[_wlOhlcv.length - 1];
            var close   = last.close;
            var dayHigh = last.high;
            var dayLow  = last.low;
            var prevBar = _wlOhlcv.length >= 2 ? _wlOhlcv[_wlOhlcv.length - 2] : null;
            var chg     = prevBar ? close - prevBar.close : 0;
            var pct     = prevBar && prevBar.close ? (chg / prevBar.close) * 100 : 0;
            var sliceLen = tf === 'W' ? 52 : tf === 'M' ? 12 : 252;
            var slice   = _wlOhlcv.slice(Math.max(0, _wlOhlcv.length - sliceLen));
            var yrLow   = slice.reduce(function(m, b) { return Math.min(m, b.low);  }, Infinity);
            var yrHigh  = slice.reduce(function(m, b) { return Math.max(m, b.high); }, -Infinity);
            var chgColor = chg >= 0 ? 'var(--success)' : 'var(--danger)';
            var chgSign  = chg >= 0 ? '+' : '';
            var barLabel = tf === 'W' ? 'WK' : tf === 'M' ? 'MO' : 'DAY';
            var barColor = chg >= 0 ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
            function mkBar(low, high, curr, width, crLabel) {
                var pos = (high > low) ? Math.max(2, Math.min(98, (curr - low) / (high - low) * 100)) : 50;
                var p = pos.toFixed(1);
                var crSpan = crLabel != null
                    ? '<span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-150%);font-size:9px;font-weight:700;color:' + crLabel.color + ';letter-spacing:.02em;pointer-events:none;">' + crLabel.text + '</span>'
                    : '';
                return '<span style="position:relative;display:inline-block;width:' + width + 'px;height:4px;border-radius:2px;background:var(--bg-surface);vertical-align:middle;flex-shrink:0;overflow:visible;">'
                    + '<span style="position:absolute;left:0;top:0;height:100%;width:' + p + '%;background:' + barColor + ';border-radius:2px;"></span>'
                    + '<span style="position:absolute;top:50%;left:' + p + '%;transform:translate(-50%,-50%);width:8px;height:8px;background:var(--text-primary-alt);border-radius:50%;box-shadow:0 0 0 1.5px var(--bg-page);"></span>'
                    + crSpan + '</span>';
            }
            var crRaw   = (dayHigh > dayLow) ? Math.round((close - dayLow) / (dayHigh - dayLow) * 100) : null;
            var crLabel = crRaw != null ? { text: crRaw + '%', color: crRaw >= 60 ? 'var(--success)' : crRaw >= 30 ? 'var(--warning-alt)' : 'var(--danger)' } : null;
            var adrEl = document.getElementById('wl-chart-mkt-adr');
            if (adrEl) {
                var adrSd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var adrRaw = adrSd ? adrSd.adr_pct : null;
                if (adrRaw != null) {
                    adrEl.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">ADR%</span>'
                                    + '<span style="color:var(--text-primary-alt);font-size:12px;">' + adrRaw.toFixed(1) + '%</span>';
                    adrEl.style.display = 'inline-flex';
                } else {
                    adrEl.style.display = 'none';
                }
            }
            var mcapEl = document.getElementById('wl-chart-mkt-mcap');
            if (mcapEl) {
                var sd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var mcapRaw = sd ? sd.MarketCap : null;
                if (mcapRaw != null) {
                    var mc = mcapRaw >= 1e12 ? (mcapRaw/1e12).toFixed(2)+'T' : mcapRaw >= 1e9 ? (mcapRaw/1e9).toFixed(2)+'B' : mcapRaw >= 1e6 ? (mcapRaw/1e6).toFixed(0)+'M' : mcapRaw;
                    mcapEl.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">Mkt Cap</span><span style="color:var(--text-primary-alt);font-size:12px;">' + mc + '</span>';
                    mcapEl.style.display = 'inline-flex';
                } else { mcapEl.style.display = 'none'; }
            }
            document.getElementById('wl-chart-mkt-price').innerHTML =
                '<span style="color:var(--text-emphasis-2);font-size:20px;font-weight:700;">' + fp(close) + '</span>' +
                '&nbsp;<span style="color:' + chgColor + ';font-size:13px;font-weight:600;">' + chgSign + fp(chg) + '&nbsp;(' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)</span>';
            document.getElementById('wl-chart-mkt-day').innerHTML =
                '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">' + barLabel + '</span>' +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(dayLow) + '</span>' +
                mkBar(dayLow, dayHigh, close, 130, crLabel) +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(dayHigh) + '</span>';
            var w52HiPct   = (yrHigh > 0) ? (yrHigh - close) / yrHigh * 100 : 0;
            var w52HiLabel = yrHigh > 0 ? {
                text:  w52HiPct < 0.5 ? 'ATH' : ('-' + w52HiPct.toFixed(1) + '%'),
                color: w52HiPct <= 5 ? 'var(--success)' : w52HiPct <= 15 ? 'var(--warning-alt)' : 'var(--danger)'
            } : null;
            document.getElementById('wl-chart-mkt-52w').innerHTML =
                '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">52W</span>' +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(yrLow) + '</span>' +
                mkBar(yrLow, yrHigh, close, 120, w52HiLabel) +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(yrHigh) + '</span>';
            document.getElementById('wl-chart-mkt-info').style.display = 'flex';
        })();

        // Keyboard: Delete/Escape for trendlines + AVWAP
        if (_wlKeyHandler) { document.removeEventListener('keydown', _wlKeyHandler); }
        _wlKeyHandler = function(evt) {
            // The fullscreen chart sits on top of this panel; its own handler owns the keyboard while it's open.
            // Without this, a letter typed in fullscreen also opened THIS panel's symbol box behind the overlay
            // (which then auto-confirmed and switched the watchlist chart to a random ticker), and Alt+D/T/A,
            // Delete and Escape reached the watchlist chart too. Same guard as _alKeyHandler in alerts.js.
            if (document.getElementById('mc-fullscreen-overlay').classList.contains('open')) return;
            if (
                evt.key.length === 1 && /[a-zA-Z0-9]/.test(evt.key) &&
                !evt.ctrlKey && !evt.metaKey && !evt.altKey &&
                evt.target.tagName !== 'INPUT' && evt.target.tagName !== 'TEXTAREA' &&
                !document.getElementById('wl-chart-sym-input')
            ) {
                window._wlChartSymClick();
                var _quickInp = document.getElementById('wl-chart-sym-input');
                if (_quickInp) {
                    _quickInp.value = evt.key.toUpperCase();
                    _quickInp.dispatchEvent(new Event('input'));
                }
                evt.preventDefault();
                return;
            }
            if (evt.key === 'Escape') {
                if (_wlMeasureActive || _wlMeasurePhase === 1) {
                    _wlMeasureActive = false;
                    _wlMeasurePhase  = 0;
                    if (_wlMeasureRafId) { cancelAnimationFrame(_wlMeasureRafId); _wlMeasureRafId = null; }
                    document.removeEventListener('mousemove', _onWlMeasureDragMove);
                    document.removeEventListener('mouseup',   _onWlMeasureDragEnd);
                    document.removeEventListener('mousemove', _onWlMeasurePreviewMove);
                }
                if (_wlMeasureResult) {
                    _hideMeasureOverlay(_wlMeasureSvgOverlay, _wlMeasureInfoDiv);
                    _wlMeasureResult = null;
                }
                if (_wlTrendDraw.active) {
                    _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
                    if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
                } else if (_wlSelectedTrendlineIdx !== -1) {
                    _wlDeselectAllTrendlines();
                } else if (_wlSelectedVwapIdx !== -1) {
                    _wlDeselectAllVwaps();
                }
                return;
            }
            // Alt shortcuts: D = tooltip, T = trendline, A = AVWAP
            if (evt.altKey && !evt.ctrlKey && !evt.metaKey) {
                if (evt.key === 'd' || evt.key === 'D') { evt.preventDefault(); window.wlToggleTooltip(); return; }
                if (evt.key === 't' || evt.key === 'T') { evt.preventDefault(); window.wlChartToggleTrendline(); return; }
                if (evt.key === 'a' || evt.key === 'A') { evt.preventDefault(); window.wlChartToggleVwap(); return; }
            }
            if (evt.key !== 'Delete') return;
            if (_measureDeleteSelected(_wlMeasureList)) { evt.preventDefault(); return; }
            if (_wlSelectedTrendlineIdx !== -1) {
                var selTl = _wlTrendlines[_wlSelectedTrendlineIdx];
                _wlSelectedTrendlineIdx = -1;
                if (selTl) _deleteTrendlineWithAlerts(_wlSym, selTl, function() {
                    var ti = _wlTrendlines.indexOf(selTl);
                    if (ti !== -1) _wlTrendlines.splice(ti, 1);
                    try { if (_wlCandle) _wlCandle.detachPrimitive(selTl.primitive); } catch(e) {}
                });
                return;
            }
            if (_wlSelectedVwapIdx !== -1) {
                var selVwap = _wlVwapSeries[_wlSelectedVwapIdx];
                _wlSelectedVwapIdx = -1;
                if (selVwap) _deleteVwapWithAlerts(_wlSym, _wlOhlcv, _wlTf, selVwap, function() {
                    var vi = _wlVwapSeries.indexOf(selVwap);
                    if (vi !== -1) _wlVwapSeries.splice(vi, 1);
                    try { _wlChart.removeSeries(selVwap.series); } catch(e) {}
                    _wlVwapSeries.forEach(function(entry) { _vwapSetSelectedLook(entry, false); });
                });
                return;
            }
            if (_wlTrendlineMode && _wlTrendlines.length) {
                var tLast = _wlTrendlines[_wlTrendlines.length - 1];
                _deleteTrendlineWithAlerts(_wlSym, tLast, function() {
                    var li = _wlTrendlines.indexOf(tLast);
                    if (li !== -1) _wlTrendlines.splice(li, 1);
                    try { if (_wlCandle) _wlCandle.detachPrimitive(tLast.primitive); } catch(e) {}
                });
            }
        };
        document.addEventListener('keydown', _wlKeyHandler);

        // Tooltip button (injected once, idempotent)
        (function() {
            var avwapBtn = document.getElementById('wl-chart-vwap-btn');
            if (avwapBtn && !document.getElementById('wl-chart-tooltip-btn')) {
                var ttBtn = document.createElement('button');
                ttBtn.id        = 'wl-chart-tooltip-btn';
                ttBtn.className = avwapBtn.className.replace(/\bactive\b/g, '').trim();
                ttBtn.title     = 'Data Tooltip (Alt+D)';
                ttBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><line x1="6" y1="1" x2="6" y2="11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><line x1="1" y1="6" x2="11" y2="6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
                ttBtn.addEventListener('click', window.wlToggleTooltip);
                avwapBtn.parentNode.insertBefore(ttBtn, avwapBtn.nextSibling);
            }
            var existing = document.getElementById('wl-chart-tooltip-btn');
            if (existing) existing.classList.toggle('active', _wlTooltipEnabled);
        })();

        // Inject today's live bar so the WL chart always shows the latest
        // intraday OHLC — mirrors the fullscreen chart fix above.
        _injectChartLiveBar(sym, tf, _wlCandle, _wlVol, _wlOhlcv,
            function() { return _wlSym !== sym || !_wlCandle; });

        // Restore alert-backed trendlines and AVWAPs so they're visible when reviewing the chart
        _restoreAlertLines(sym, tf, ohlcv, _addWlTrendline, _addWlVwap);
        // Then the lines you drew by hand (see the fullscreen chart above).
        (function() {
            var chartRef = _wlChart;
            _cdRestore(sym, tf, {
                isStale:        function() { return _wlChart !== chartRef || _wlSym !== sym || !_wlCandle; },
                getOhlcv:       function() { return _wlOhlcv; },
                getTrendlines:  function() { return _wlTrendlines; },
                getVwapAnchors: function() { return _wlVwapSeries.map(function(v) { return v.anchor; }); },
                addTrendline:   _addWlTrendline,
                addVwap:        _addWlVwap
            });
        })();
        // The measurements you committed on this ticker (by time + price, so any timeframe).
        (function() {
            var chartRef = _wlChart;
            _cdRestoreMs(sym, {
                isStale:     function() { return _wlChart !== chartRef || _wlSym !== sym || !_wlCandle; },
                getOhlcv:    function() { return _wlOhlcv; },
                getMeasures: function() { return _wlMeasureList; },
                addMeasure:  function(a, b) {
                    var ms = _measureCommit({ contRef: _wlTrendContRef, measureList: _wlMeasureList }, a.time, a.price, b.time, b.price);
                    ms.sym = sym;
                },
                render:      function() { _measureRenderAll(_wlMeasureList, _wlChart, _wlCandle, _wlTrendContRef, _wlOhlcv); }
            });
        })();
    }

    // ── WL chart controls (exposed to HTML onclick) ───────────────────────
    window.wlChartSetTf = function(tf) {
        if (!_wlSym) return;
        _wlTf = tf;
        document.querySelectorAll('.wl-chart-fs-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        _wlVwapMode = false; _wlVwapSeries = []; _wlSelectedVwapIdx = -1;
        var vwapBtn = document.getElementById('wl-chart-vwap-btn');
        if (vwapBtn) vwapBtn.classList.remove('active');
        _wlTrendlines = []; _wlTrendlineFirst = null; _wlSelectedTrendlineIdx = -1;
        if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
        _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
        _wlMeasureMode = false; _wlMeasureActive = false; _wlMeasurePhase = 0; _wlMeasureResult = null;
        if (_wlMeasureRafId) { cancelAnimationFrame(_wlMeasureRafId); _wlMeasureRafId = null; }
        var wlMBtn = document.getElementById('wl-chart-measure-btn');
        if (wlMBtn) wlMBtn.classList.remove('active');
        document.removeEventListener('mousemove', _onWlMeasureDragMove);
        document.removeEventListener('mouseup',   _onWlMeasureDragEnd);
        document.removeEventListener('mousemove', _onWlMeasurePreviewMove);
        var maPanel   = document.getElementById('wl-chart-ma-panel');
        var maChevron = document.getElementById('wl-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
        _wlVisibleBars = tf === 'D' ? 252 : tf === 'W' ? 104 : 60;
        // No forced cache-clear — fetchMcOhlcv now enforces a 30-min freshness
        // window centrally (see MC_CACHE_TTL_MS), so switching back to a TF
        // already fetched recently serves instantly instead of re-entering
        // the queue, and still gets a real refetch periodically.
        var sym = _wlSym;
        var container = document.getElementById('wl-chart-widget');
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;">Loading\u2026</div>';
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (_wlSym !== sym || _wlTf !== tf) return;
            _buildWlChart(sym, ohlcv, tf);
        });
    };

    window.wlChartToggleMaPanel = function(e) {
        e.stopPropagation();
        var panel   = document.getElementById('wl-chart-ma-panel');
        var chevron = document.getElementById('wl-chart-ma-chevron');
        if (!panel) return;
        var opening = panel.style.display === 'none';
        panel.style.display = opening ? '' : 'none';
        if (chevron) chevron.style.transform = opening ? 'rotate(180deg)' : '';
        if (opening) {
            setTimeout(function() {
                function _outsideClick(ev) {
                    var wrap = document.getElementById('wl-chart-ma-wrap');
                    if (wrap && !wrap.contains(ev.target)) {
                        panel.style.display = 'none';
                        if (chevron) chevron.style.transform = '';
                        document.removeEventListener('click', _outsideClick, true);
                    }
                }
                document.addEventListener('click', _outsideClick, true);
            }, 0);
        }
    };

    window.wlChartToggleMa = function(key) {
        _wlActiveMas[key] = !_wlActiveMas[key];
        var btn = document.getElementById('wl-chart-ma-' + key);
        if (btn) btn.classList.toggle('active', _wlActiveMas[key]);
        if (!_wlChart || !_wlOhlcv.length) return;
        if (_wlActiveMas[key]) {
            if (_wlMaSeries[key]) return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = _wlChart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
            var maData = _calcMA(_wlOhlcv, key);
            s.setData(maData);
            _wlMaSeries[key]  = s;
            _wlMaDataMap[key] = new Map(maData.map(function(d) { return [d.time, d.value]; }));
        } else {
            if (_wlMaSeries[key]) { try { _wlChart.removeSeries(_wlMaSeries[key]); } catch(e) {} delete _wlMaSeries[key]; }
            delete _wlMaDataMap[key];
        }
    };

    window.wlChartToggleVwap = function() {
        _wlVwapMode = !_wlVwapMode;
        var btn = document.getElementById('wl-chart-vwap-btn');
        if (btn) btn.classList.toggle('active', _wlVwapMode);
        if (_wlVwapMode && _wlTrendlineMode) {
            _wlTrendlineMode = false;
            var tBtn = document.getElementById('wl-chart-trendline-btn');
            if (tBtn) tBtn.classList.remove('active');
            _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
            if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
        }
        if (_wlVwapMode && _wlMeasureMode) {
            _wlMeasureMode = false;
            var mBtn = document.getElementById('wl-chart-measure-btn');
            if (mBtn) mBtn.classList.remove('active');
        }
    };

    window.wlChartToggleTrendline = function() {
        _wlTlMenu.resetStyle(); // plain click and Alt+T always draw solid
        _wlTrendlineMode = !_wlTrendlineMode;
        var btn = document.getElementById('wl-chart-trendline-btn');
        if (btn) btn.classList.toggle('active', _wlTrendlineMode);
        if (_wlTrendlineMode && _wlVwapMode) {
            _wlVwapMode = false;
            var vBtn = document.getElementById('wl-chart-vwap-btn');
            if (vBtn) vBtn.classList.remove('active');
        }
        if (_wlTrendlineMode && _wlMeasureMode) {
            _wlMeasureMode = false;
            var mBtn = document.getElementById('wl-chart-measure-btn');
            if (mBtn) mBtn.classList.remove('active');
        }
        _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
        _wlTrendlineFirst = null;
        if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
        if (_wlSelectedTrendlineIdx !== -1) _wlDeselectAllTrendlines();
    };

    // Watchlist chart instance of the shared hold-menu (see _makeTrendlineStyleHold) — #wl-chart-trendline-btn
    var _wlTlMenu = _makeTrendlineStyleHold({
        toggle:     function() { window.wlChartToggleTrendline(); },
        isActive:   function() { return _wlTrendlineMode; },
        cancelDraw: function() {
            _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
            if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
        }
    });
    window.wlChartTlBtnDown  = _wlTlMenu.down;
    window.wlChartTlBtnClick = _wlTlMenu.click;

    window.wlChartToggleMeasure = function() {
        _wlMeasureMode = !_wlMeasureMode;
        var btn = document.getElementById('wl-chart-measure-btn');
        if (btn) btn.classList.toggle('active', _wlMeasureMode);
        if (_wlMeasureMode) {
            if (_wlTrendlineMode) {
                _wlTrendlineMode = false;
                var tBtn = document.getElementById('wl-chart-trendline-btn');
                if (tBtn) tBtn.classList.remove('active');
                _wlTrendDraw.active = false; _wlTrendDraw.startTime = null; _wlTrendDraw.startPrice = null;
                if (_wlTrendSvgOverlay) _wlTrendSvgOverlay.style.display = 'none';
            }
            if (_wlVwapMode) {
                _wlVwapMode = false;
                var vBtn = document.getElementById('wl-chart-vwap-btn');
                if (vBtn) vBtn.classList.remove('active');
            }
        } else {
            if (_wlMeasureActive || _wlMeasurePhase === 1) {
                _wlMeasureActive = false;
                _wlMeasurePhase  = 0;
                if (_wlMeasureRafId) { cancelAnimationFrame(_wlMeasureRafId); _wlMeasureRafId = null; }
                document.removeEventListener('mousemove', _onWlMeasureDragMove);
                document.removeEventListener('mouseup',   _onWlMeasureDragEnd);
                document.removeEventListener('mousemove', _onWlMeasurePreviewMove);
            }
            _hideMeasureOverlay(_wlMeasureSvgOverlay, _wlMeasureInfoDiv);
            _wlMeasureResult = null;
        }
    };

    window.wlToggleTooltip = function() {
        _wlTooltipEnabled = !_wlTooltipEnabled;
        var btn = document.getElementById('wl-chart-tooltip-btn');
        if (btn) btn.classList.toggle('active', _wlTooltipEnabled);
        if (!_wlTooltipEnabled && _lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
    };

    // ══════════════════════════════════════════════════════════════════════

    window.filterStocksTable = function(q) {
        q = (q || '').toLowerCase();
        document.querySelectorAll('#stocks-tbody .stock-row').forEach(function(row) {
            var sym = (row.getAttribute('data-symbol') || '').toLowerCase();
            row.style.display = (!q || sym.includes(q)) ? '' : 'none';
        });
    };



    window.goToSector = function(sector) {
        showView('industries');
        setIndView('heatmap');
        searchQuery = sector;
        var si = document.getElementById('search-input');
        if (si) { si.value = sector; _updateSearchClear(sector); }
        renderHeatmap();
    };

    window.navToIndustries = function() {
        if (currentView === 'industry-stocks') {
            backToIndustries();
        } else if (_lastIndustryName) {
            openIndustry(_lastIndustryName);
            var savedScroll = _lastIndustryScrollTop;
            if (savedScroll > 0) {
                setTimeout(function() {
                    var wrap = document.querySelector('#view-industry-stocks .stocks-table-wrap');
                    if (wrap) wrap.scrollTop = savedScroll;
                }, 0);
            }
        } else {
            showView('industries');
        }
    };

    window.backToIndustries = function() {
        _lastIndustryName      = null;
        _lastIndustryScrollTop = 0;
        multichartActive = false;
        mcTickers = [];
        mcWidgets = {};
        document.getElementById('stocks-table-view').style.display      = 'flex';
        document.getElementById('stocks-multichart-view').style.display = 'none';
        document.getElementById('multichart-toggle-btn').style.background  = '';
        document.getElementById('multichart-toggle-btn').style.borderColor = '';
        document.getElementById('multichart-toggle-btn').style.color       = '';
        indStopPricePolling();
        showView('industries');
        var _savedIndListScroll = _industriesListScrollTop;
        if (_savedIndListScroll > 0) {
            setTimeout(function() {
                var _mainArea = document.getElementById('main-area');
                if (_mainArea) _mainArea.scrollTop = _savedIndListScroll;
            }, 0);
        }
    };

