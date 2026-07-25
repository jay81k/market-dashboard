
    var multichartActive = false;
    var mcTimeframe      = 'D';
    var mcCols           = parseInt(localStorage.getItem('mcSharedCols') || '4');
    var mcTickers        = [];
    var mcWidgets        = {};

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
        document.getElementById('multichart-toggle-btn').style.background = multichartActive ? '#1f3a5c' : '';
        document.getElementById('multichart-toggle-btn').style.borderColor = multichartActive ? '#388bfd' : '';
        document.getElementById('multichart-toggle-btn').style.color = multichartActive ? '#58a6ff' : '';
        if (multichartActive) renderMulticharts();
    };

    window.setMcTf = function(tf) {
        mcTimeframe = tf;
        document.querySelectorAll('#stocks-multichart-view .mc-tf-btn').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-tf') === tf); });
        renderMulticharts();
    };

    // ── LW Multichart Infrastructure ─────────────────────────────────────────

    var _mcOhlcvCache   = {};   // { "AAPL_D": [...ohlcv] }
    var _mcMetaCache    = {};   // { "AAPL": { marketState, preMarketPrice, postMarketPrice, ... } }
    var _mcFetchQueue   = {};   // per-tf: { pending:[], active:0, resolvers:{} }
    var MC_FETCH_LIMIT  = 12; // was 6, originally 50 — pre-warm-all is gone and this cap plus the cooldown below are now the real safety net, so this can sit closer to what's actually visible (~12 tiles) instead of being deliberately conservative

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
    var _TRENDLINE_COLOR            = '#ffffff';
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

    // Measure tool state (wl)
    var _wlMeasureMode       = false;
    var _wlMeasureActive     = false;
    var _wlMeasurePhase      = 0;
    var _wlMeasureRafId      = null;
    var _wlMeasureStart      = null;
    var _wlMeasureResult     = null;
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
        SMA200: { period: 200, color: '#2e2e2e', ema: false },
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
        return tf === 'W' ? 'max' : tf === 'M' ? 'max' : '10y';
    }

    // Concurrent fetch queue — max MC_FETCH_LIMIT in-flight at once
    function fetchMcOhlcv(sym, tf, gridMode) {
        var key = sym + '_' + tf + (gridMode ? '_grid' : '');
        if (_mcOhlcvCache[key] !== undefined) return Promise.resolve(_mcOhlcvCache[key]);
        return new Promise(function(resolve) {
            var qKey = tf + (gridMode ? '_grid' : '');
            if (!_mcFetchQueue[qKey]) _mcFetchQueue[qKey] = { pending: [], active: 0, resolvers: {} };
            var q = _mcFetchQueue[qKey];
            if (!q.resolvers[sym]) q.resolvers[sym] = [];
            q.resolvers[sym].push(resolve);
            if (q.pending.indexOf(sym) === -1) q.pending.push(sym);
            _drainMcQueue(tf, gridMode);
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
            if (_mcOhlcvCache[key] !== undefined) {
                q.pending.shift();
                var res0 = (q.resolvers[sym] || []).splice(0);
                delete q.resolvers[sym];
                res0.forEach(function(r) { r(_mcOhlcvCache[key]); });
                continue;
            }

            var sinceLast = Date.now() - window.yahooProxyPace.lastLaunchAt();
            if (sinceLast < MC_LAUNCH_MIN_SPACING) {
                if (!q._paceTimer) {
                    q._paceTimer = setTimeout(function() {
                        q._paceTimer = null;
                        _drainMcQueue(tf, gridMode);
                    }, MC_LAUNCH_MIN_SPACING - sinceLast);
                }
                break;
            }

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
                        // Strip incomplete current period (Yahoo includes it, TV doesn't)
                        if (ohlcv.length > 0 && tf !== 'D') {
                            var now      = new Date();
                            var lastDate = new Date(ohlcv[ohlcv.length - 1].time * 1000);
                            if (tf === 'M') {
                                if (lastDate.getUTCMonth() === now.getUTCMonth() && lastDate.getUTCFullYear() === now.getUTCFullYear()) ohlcv.pop();
                            } else if (tf === 'W') {
                                var dow = now.getUTCDay();
                                var daysSinceMon = (dow + 6) % 7;
                                var lastMon = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() - daysSinceMon));
                                if (lastDate >= lastMon) ohlcv.pop();
                            }
                        }
                        _mcOhlcvCache[s + '_' + tf + (gridMode ? '_grid' : '')] = ohlcv;
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
                            // q.active stays held during the wait so the slot isn't reused
                            setTimeout(function() { doFetch(s, attempt + 1); }, delayMs);
                        } else {
                            // Don't cache this as confirmed-empty data — it's a failure, not
                            // "no data for this ticker." Leaving the cache key unset means a
                            // later retry (e.g. clicking the failed tile) triggers a real
                            // fetch instead of instantly resolving to a stale permanent blank.
                            var res = (q.resolvers[s] || []).splice(0);
                            delete q.resolvers[s];
                            res.forEach(function(r) { r(null); }); // null = failed, distinct from [] = genuinely no data
                            q.active = Math.max(0, q.active - 1);
                            _drainMcQueue(tf, gridMode);
                        }
                    });
            })(sym, 0);
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
    function _calcAVWAP(ohlcv, anchorIdx) {
        var out = [], cumVT = 0, cumV = 0;
        for (var i = anchorIdx; i < ohlcv.length; i++) {
            var tp = (ohlcv[i].open + ohlcv[i].high + ohlcv[i].low + ohlcv[i].close) / 4;
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
                'background:rgba(13,17,23,0.96);border:1px solid #30363d;border-radius:5px;' +
                'padding:8px 12px;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums;' +
                'font-family:inherit;color:#c9d1d9;line-height:1.75;white-space:nowrap;' +
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
        var cl     = d.close >= d.open ? '#089981' : '#b22833';
        var delta  = 0, pct = 0, chgClr = '#6e7681';
        if (barIdx > 0) {
            var prevClose = ohlcv[barIdx - 1].close;
            delta  = d.close - prevClose;
            pct    = (delta / prevClose) * 100;
            chgClr = delta >= 0 ? '#3fb950' : '#f85149';
        }
        var cr    = (d.high > d.low) ? Math.round((d.close - d.low) / (d.high - d.low) * 100) : null;
        var crClr = cr != null ? (cr >= 60 ? '#3fb950' : cr >= 30 ? '#e3852b' : '#f85149') : '#6e7681';
        var vol   = ohlcv[barIdx] ? ohlcv[barIdx].volume : null;
        var L = '<span style="color:#6e7681">', V = '<span style="color:#c9d1d9">', E = '</span>';
        var html = '<div style="color:#8b949e;margin-bottom:3px;">' + _fmtBarDate(barTime) + '</div>';
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
                var vpClr = vp >= 0 ? '#3fb950' : '#f85149';
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
            var dpClr = dp >= 0 ? '#3fb950' : '#f85149';
            maRows.push(
                '<span style="color:' + def.color + '">' + label + '</span>' +
                '<span style="color:#c9d1d9;justify-self:end">' + fp(maVal) + '</span>' +
                '<span style="color:' + dpClr + ';justify-self:end">' + (dp >= 0 ? '+' : '') + dp.toFixed(1) + '%</span>'
            );
        });
        if (maRows.length) {
            html += '<div style="border-top:1px solid #30363d;margin:5px 0 4px;"></div>';
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
            info.style.cssText = 'position:absolute;display:none;z-index:7;pointer-events:none;color:#fff;font-size:11.5px;font-weight:600;font-family:inherit;font-variant-numeric:tabular-nums;line-height:1.55;padding:5px 9px;border-radius:3px;white-space:nowrap;';
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

        var fillClr   = isUp ? 'rgba(8,153,129,0.15)'  : 'rgba(178,40,51,0.18)';
        var strokeClr = isUp ? 'rgba(8,153,129,0.55)'  : 'rgba(178,40,51,0.6)';
        var infoBg    = isUp ? 'rgba(8,153,129,0.88)'  : 'rgba(178,40,51,0.88)';

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
        hLineEl.setAttribute('stroke', 'rgba(255,255,255,0.22)');
        hLineEl.setAttribute('stroke-width', '0.5');
        hLineEl.setAttribute('stroke-dasharray', '3,3');

        var dStr = (result.priceDelta >= 0 ? '+' : '') + result.priceDelta.toFixed(2);
        var pStr = (result.pctDelta   >= 0 ? '+' : '') + result.pctDelta.toFixed(2)   + '%';
        infoEl.innerHTML = '<div>' + dStr + ' (' + pStr + ')</div><div>' + result.barCount + ' bars, ' + result.dayCount + 'd</div>';
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

    function _computeMeasureResult(ohlcv, startTime, startPrice, endTime, endPrice) {
        // bar count = |endIdx - startIdx| (TV-style: intervals between bars)
        var si = _barIdxByTime(ohlcv, startTime);
        var ei = _barIdxByTime(ohlcv, endTime);
        if (si === -1) si = ohlcv.length - 1;
        if (ei === -1) ei = ohlcv.length - 1;
        var barCount = Math.abs(ei - si);
        // calendar days from unix second timestamps
        var dayCount = Math.round(Math.abs(endTime - startTime) / 86400);
        if (dayCount === 0 && barCount > 0) dayCount = barCount; // fallback
        var priceDelta = endPrice - startPrice;
        var pctDelta   = (priceDelta / Math.abs(startPrice)) * 100;
        return { startTime: startTime, startPrice: startPrice,
                 endTime: endTime, endPrice: endPrice,
                 startBarIdx: si, endBarIdx: ei,
                 barCount: barCount, dayCount: dayCount,
                 priceDelta: priceDelta, pctDelta: pctDelta };
    }

    function _measureGetTimeAtX(chart, ohlcv, lx) {
        var t = chart.timeScale().coordinateToTime(lx);
        if (t != null) return t;
        // Past last bar — extrapolate
        var last = ohlcv[ohlcv.length - 1];
        var prev = ohlcv.length > 1 ? ohlcv[ohlcv.length - 2] : last;
        var lastX = chart.timeScale().timeToCoordinate(last.time);
        var prevX = chart.timeScale().timeToCoordinate(prev.time);
        var pxPer = (lastX != null && prevX != null) ? Math.abs(lastX - prevX) : 8;
        var barSec = last.time - prev.time;
        var ahead  = Math.max(1, Math.round((lx - lastX) / pxPer));
        return last.time + ahead * barSec;
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── Shared cell chart renderer ────────────────────────────────────────────
    function renderLwMcCellChart(container, ohlcv) {
        container.innerHTML = '';
        if (!window.LightweightCharts || !ohlcv || !ohlcv.length) {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:11px;">No data</div>';
            return null;
        }
        var chart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: { background: { color: '#0d1117' }, textColor: '#6e7681' },
            grid:    { vertLines: { color: '#171b22' }, horzLines: { color: '#171b22' } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
            rightPriceScale: { borderColor: '#21262d', textColor: '#6e7681', scaleMargins: { top: 0.06, bottom: 0.22 } },
            timeScale: { borderColor: '#21262d', timeVisible: false, rightOffset: 1 },
            handleScroll: false, handleScale: false,
        });
        var candle = chart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: '#089981', downColor: '#b22833', borderVisible: false,
            wickUpColor: '#089981', wickDownColor: '#b22833',
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
            return { time: d.time, value: d.volume, color: d.close >= d.open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)' };
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
        leg.style.cssText = 'position:absolute;top:0;left:6px;z-index:10;font-size:10px;font-weight:600;font-variant-numeric:tabular-nums;color:#8b949e;pointer-events:none;line-height:1.5;background:rgba(13,17,23,0.7);padding:2px 5px;border-radius:3px;';
        container.style.position = 'relative';
        container.appendChild(leg);
        function fp(v) { return v != null ? v.toFixed(2) : '—'; }
        function fv(v) { return v==null?'—':v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':v.toFixed(0); }
        chart.subscribeCrosshairMove(function(p) {
            if (!p.time || !p.seriesData || !p.seriesData.size) { leg.innerHTML = ''; return; }
            var d = p.seriesData.get(candle); if (!d) { leg.innerHTML = ''; return; }
            var cl = d.close >= d.open ? '#089981' : '#b22833';
            var vd = p.seriesData.get(vol);
            leg.innerHTML = '<span style="color:#6e7681">O</span><span style="color:'+cl+'">'+fp(d.open)+'</span> <span style="color:#6e7681">H</span><span style="color:'+cl+'">'+fp(d.high)+'</span> <span style="color:#6e7681">L</span><span style="color:'+cl+'">'+fp(d.low)+'</span> <span style="color:#6e7681">C</span><span style="color:'+cl+'">'+fp(d.close)+'</span>'+(vd?'  <span style="color:#484f58">V</span><span style="color:#6e7681">'+fv(vd.value)+'</span>':'');
        });
        return { chart: chart, candle: candle, vol: vol, ohlcv: ohlcv };
    }

    // ── Push live intraday price into a rendered multichart cell ──────────────
    function _updateMcLiveCandle(ticker, price, dayHigh, dayLow, widgetsObj) {
        var inst = widgetsObj && widgetsObj[ticker];
        if (!inst || !inst.candle || !inst.ohlcv || !inst.ohlcv.length) return;
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
            // appearing after close when W/M strips the current period bar.
            if (!wlIsMarketOpen()) return;
            open = high = low = price;
            volume = 0;
            inst.ohlcv.push({ time: todayTs, open: open, high: high, low: low, close: price, volume: volume });
        }
        try { inst.candle.update({ time: todayTs, open: open, high: high, low: low, close: price, volume: volume }); } catch(e) {}
        if (inst.vol) {
            try { inst.vol.update({ time: todayTs, value: volume, color: price >= open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)' }); } catch(e) {}
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
            chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:11px;">Loading…</div>';

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
                chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:11px;">Loading…</div>';
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
                        chartDiv.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:4px;color:#8b949e;font-size:11px;"><span>Failed to load</span><span style="text-decoration:underline;">Click to retry</span></div>';
                        return;
                    }
                    try {
                        var inst = renderLwMcCellChart(chartDiv, ohlcv);
                        rendered = true;
                        if (inst) widgetsObj[sym] = inst;
                    } catch(e) {
                        chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:11px;">Error</div>';
                    }
                }).catch(function() {
                    chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:11px;">Error</div>';
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
    function _addFsVwap(anchorIdx) {
        if (!_mcFsChart || !_mcFsOhlcv.length) return;
        var color = _AVWAP_COLOR;
        var data  = _calcAVWAP(_mcFsOhlcv, anchorIdx);
        if (!data.length) return;
        var s = _mcFsChart.addSeries(LightweightCharts.LineSeries, { color: color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
        s.setData(data);
        var dataMap = new Map(data.map(function(d) { return [d.time, d.value]; }));
        _mcFsVwapSeries.push({ series: s, anchor: anchorIdx, color: color, dataMap: dataMap });
    }

    // Converts a time to a pixel X coordinate, extrapolating linearly for
    // future timestamps that have no entry in LWC's internal time scale.
    function _mcFsTimeToX(chart, ohlcv, time) {
        var x = chart.timeScale().timeToCoordinate(time);
        if (x !== null) return x;
        if (ohlcv.length < 2) return null;
        var last  = ohlcv[ohlcv.length - 1];
        var prev  = ohlcv[ohlcv.length - 2];
        var lastX = chart.timeScale().timeToCoordinate(last.time);
        var prevX = chart.timeScale().timeToCoordinate(prev.time);
        if (lastX == null || prevX == null) return null;
        var pxPerSec = (lastX - prevX) / (last.time - prev.time);
        return lastX + pxPerSec * (time - last.time);
    }

    // Draw a trendline through two bar-index/price anchors using a v5 canvas
    // primitive.  The trendline object stores leftP/rightP for hit-testing and
    // a `selected` flag that the renderer reads to draw the highlight state.
    // Builds the canvas-primitive trendline object and attaches it to the
    // candle series. Shared by fullscreen/watchlist/alerts (alerts.js calls
    // this cross-file, same pattern as the other shared cores above).
    function _addTrendlineCore(p1, p2, chart, candle, ohlcv, trendlines) {
        if (!chart || !candle || !ohlcv.length) return;
        var refChart  = chart;
        var refSeries = candle;

        // Normalise so leftP is always the earlier anchor (by time, supports future timestamps)
        var leftP  = p1.time <= p2.time ? p1 : p2;
        var rightP = p1.time <= p2.time ? p2 : p1;

        // Create the object first so the primitive closes over it
        var tlObj = { p1: p1, p2: p2, leftP: leftP, rightP: rightP, selected: false, requestUpdate: null };

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
                                    ctx.strokeStyle = _TRENDLINE_COLOR;
                                    ctx.lineWidth   = 1.5 * rx;
                                    ctx.stroke();
                                    // Anchor dots only when selected
                                    if (tlObj.selected) {
                                        [[bx1, by1], [bx2, by2]].forEach(function(pt) {
                                            ctx.beginPath();
                                            ctx.arc(pt[0], pt[1], 4.5 * rx, 0, Math.PI * 2);
                                            ctx.fillStyle   = _TRENDLINE_COLOR;
                                            ctx.fill();
                                            ctx.strokeStyle = '#0d1117';
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
    }

    function _addFsTrendline(p1, p2) {
        _addTrendlineCore(p1, p2, _mcFsChart, _mcFsCandle, _mcFsOhlcv, _mcFsTrendlines);
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

    function _selectVwapCore(vwapSeries, idx) {
        vwapSeries.forEach(function(entry, i) {
            entry.series.applyOptions({ lineWidth: i === idx ? 3 : 1.5 });
        });
    }

    function _deselectAllVwapsCore(vwapSeries) {
        vwapSeries.forEach(function(entry) {
            entry.series.applyOptions({ lineWidth: 1.5 });
        });
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
            time = last.time + barsAhead * barSec;
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
            if (tl) { tl.dragging = false; if (tl.requestUpdate) tl.requestUpdate(); }
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

    // ── Shared trendline mousedown core — used by all three charts. Handles the
    // measure-tool intercept, anchor-drag pickup, line select/deselect, and the
    // two-click draw flow. (alerts.js calls this cross-file.)
    function _onTrendMouseDownCore(evt, cfg) {
        if (evt.button !== 0 || !cfg.candle || !cfg.chart || !cfg.contRef) return;

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
            var _mSi   = _barIdxByTime(cfg.ohlcv, _mT);

            if (cfg.getMeasurePhase() === 1) {
                // Second click — finalise at current cursor position
                var result = _computeMeasureResult(cfg.ohlcv, cfg.getMeasureStart().time, cfg.getMeasureStart().price, _mT, _mP);
                cfg.setMeasureResult(result);
                _renderMeasureOverlay(cfg.chart, cfg.candle, cfg.contRef,
                    cfg.measureSvgOverlay, cfg.measureSvgRect, cfg.measureHLine,
                    cfg.measureInfoDiv, result);
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
                time = _last.time + _barsAhead * _barSec;
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
                cfg.svgOverlay.style.display = '';
            }
        } else {
            // ── Second click: finalise ──────────────────────────────────────
            var p1 = { time: cfg.trendDraw.startTime, price: cfg.trendDraw.startPrice };
            cfg.trendDraw.active = false;
            cfg.trendDraw.startTime = null; cfg.trendDraw.startPrice = null;
            if (cfg.svgOverlay) cfg.svgOverlay.style.display = 'none';
            if (time !== p1.time) cfg.addTrendline(p1, { time: time, price: price });
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
            measurePreviewMoveHandler: _onMcFsMeasurePreviewMove,
            getSelectedIdx:    function() { return _mcFsSelectedTrendlineIdx; },
            setSelectedIdx:    function(v) { _mcFsSelectedTrendlineIdx = v; },
            trendlines:        _mcFsTrendlines,
            deselectAllTrendlines: _deselectAllTrendlines,
            anchorHitTest:     _anchorHitTest,
            trendlineHitTest:  _trendlineHitTest,
            dragMoveHandler:   _onTrendAnchorDragMove,
            dragEndHandler:    _onTrendAnchorDragEnd,
            getTrendlineMode:  function() { return _mcFsTrendlineMode; },
            setTrendlineMode:  function(v) { _mcFsTrendlineMode = v; },
            getLastCrosshairTime: function() { return _mcFsLastCrosshairTime; },
            addTrendline:      _addFsTrendline,
            doneBtnId:         'mc-fs-trendline-btn'
        });
    }

    // ── Trendline SVG preview: mousemove drives the overlay line ──────────
    // Reads raw pixel coords and uses timeToCoordinate/priceToCoordinate to
    // place the anchor — zero LW Charts canvas re-render on every move.
    // Shared trendline-draw-preview / hover-cursor core — used by all three charts
    function _onTrendMouseMoveCore(evt, cfg) {
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
            trendlineHitTest: _trendlineHitTest
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
            window.alAddTrendlineAlert(sym, tl.p1, tl.p2, direction);
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
            dismiss:         _mcFsDismissCtx
        });
    };

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
                cfg.setCtxTrendline({ p1: _tlHit.leftP, p2: _tlHit.rightP });
                cfg.setCtxPrice(null);
                cfg.setCtxMa(null);
                document.getElementById(cfg.ctxLabelId).textContent     = cfg.getSym() + ' · Trendline';
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Alert above trendline';
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Alert below trendline';
                var _tlMenu = document.getElementById(cfg.ctxMenuId);
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
                document.getElementById(cfg.ctxLabelId).textContent     = cfg.getSym() + ' · AVWAP';
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Alert above AVWAP';
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Alert below AVWAP';
                var _avMenu = document.getElementById(cfg.ctxMenuId);
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
                document.getElementById(cfg.ctxLabelId).textContent     = cfg.getSym() + ' · ' + maLabel;
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Price crosses above ' + maLabel;
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Price crosses below ' + maLabel;
            } else {
                var fmt = '$' + price.toFixed(2);
                document.getElementById(cfg.ctxLabelId).textContent     = cfg.getSym() + ' · ' + fmt;
                document.getElementById(cfg.ctxAboveTxtId).textContent  = 'Alert above ' + fmt;
                document.getElementById(cfg.ctxBelowTxtId).textContent  = 'Alert below ' + fmt;
            }
            var menu  = document.getElementById(cfg.ctxMenuId);
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
            ctxLabelId:     'mc-fs-ctx-label',
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

    // ── Live bar injection — fullscreen + WL charts ───────────────────────────
    // Yahoo's 10y/1d historical feed sometimes omits today's partial bar or
    // carries a stale snapshot of it.  This helper mirrors _updateMcLiveCandle
    // but targets a specific candle/vol/ohlcvArr triple rather than a widgets map.
    // Price is resolved from in-memory live caches (indLivePrices → wlLivePrices
    // → snapshot); if none is found a lightweight 2-day proxy fetch is issued as
    // a fallback.  Only runs for the Daily timeframe (W/M bars are always closed).
    function _injectChartLiveBar(sym, tf, candle, vol, ohlcvArr, isStale) {
        if (tf !== 'D' || !candle || !ohlcvArr || !ohlcvArr.length) return;

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

        function _applyLiveBar(p, dh, dl) {
            if (!p || !candle || !ohlcvArr.length) return;
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
                try { vol.update({ time: todayTs, value: volume, color: p >= open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)' }); } catch(e) {}
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

            badge.style.color = up ? '#3fb950' : '#f85149';
            badge.innerHTML =
                '<span style="color:#8b949e;font-weight:500;">' + label + '</span>&nbsp; ' +
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

    function _buildFsChart(sym, ohlcv, tf) {
        var container = document.getElementById('mc-fullscreen-chart');
        container.innerHTML = '';
        if (_mcFsChart) { try { _mcFsChart.remove(); } catch(e) {} _mcFsChart = null; }
        _mcFsCandle = null; _mcFsVol = null; _mcFsVolMa = null; _mcFsVolData = null; _mcFsMaSeries = {}; _mcFsVwapSeries = []; _mcFsTrendlines = []; _mcFsTrendlineFirst = null;
        _mcFsTrendSvgOverlay = null; _mcFsTrendSvgLine = null; // SVG lives inside container.innerHTML = '' above
        _mcFsTrendDraw.active = false; _mcFsTrendDraw.startTime = null; _mcFsTrendDraw.startPrice = null;
        _mcFsSelectedTrendlineIdx = -1;
        _mcFsSelectedVwapIdx = -1;
        _mcFsDismissCtx();

        _mcFsOhlcv = ohlcv || [];
        _mcFsSym   = sym;
        _mcFsTf    = tf;
        _mcFsLastCrosshairPrice = null;

        if (!window.LightweightCharts || !_mcFsOhlcv.length) {
            var _mcFsMsg = ohlcv === null ? 'Failed to load — click to retry' : 'No data';
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;' + (ohlcv === null ? 'cursor:pointer;text-decoration:underline;' : '') + '">' + _mcFsMsg + '</div>';
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
            _mcFsTrendSvgLine.setAttribute('stroke', _TRENDLINE_COLOR);
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
            layout: { background: { color: '#0d1117' }, textColor: '#6e7681', panes: { separatorColor: '#161b22', separatorHoverColor: 'rgba(33,38,45,0.5)' } },
            grid:    { vertLines: { visible: false }, horzLines: { visible: false } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: '#21262d', textColor: '#6e7681', scaleMargins: { top: 0.05, bottom: 0.02 } },
            timeScale: { borderColor: '#21262d', timeVisible: false, secondsVisible: false, rightOffset: 12 },
            handleScroll: true, handleScale: true,
        });
        _mcFsAttachCtxMenu(); // attach once, capture phase, safe to call repeatedly

        _mcFsCandle = _mcFsChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: '#089981', downColor: '#b22833', borderVisible: false,
            wickUpColor: '#089981', wickDownColor: '#b22833',
            priceLineVisible: false, lastValueVisible: true,
        });
        _mcFsCandle.setData(_mcFsOhlcv);

        _mcFsVol = _mcFsChart.addSeries(LightweightCharts.HistogramSeries, {
            color: '#63a0f8', priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: true,
        }, 1);
        _mcFsVol.setData(_mcFsOhlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)' };
        }));
        _mcFsVol.priceScale().applyOptions({
            visible: true,
            borderColor: '#21262d',
            textColor: '#6e7681',
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
            var color = volDiffPct >= 0 ? '#3fb950' : '#f85149';

            var lbl = document.createElement('div');
            lbl.id = 'mc-fs-vol-pct-label';
            lbl.style.cssText = 'position:absolute;z-index:20;pointer-events:none;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:3px;white-space:nowrap;line-height:1;';
            lbl.innerHTML = '<span style="color:#484f58;">›</span>'
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
                    if (lastX == null || lastX < 0) {
                        lbl.style.display = 'none';
                        return;
                    }
                    lbl.style.display = 'flex';
                    lbl.style.left = (lastX + 10) + 'px';
                    lbl.style.top  = lblTop;
                }

                positionVolLabel(); // initial paint
                _mcFsChart.timeScale().subscribeVisibleTimeRangeChange(positionVolLabel);
            }, 60);
        })();

        // Re-render measure overlay on pan/zoom so the rect tracks correctly
        _mcFsChart.timeScale().subscribeVisibleLogicalRangeChange(function() {
            if (_mcFsMeasureResult) {
                _renderMeasureOverlay(_mcFsChart, _mcFsCandle, _mcFsTrendContRef,
                    _mcFsMeasureSvgOverlay, _mcFsMeasureSvgRect, _mcFsMeasureHLine,
                    _mcFsMeasureInfoDiv, _mcFsMeasureResult);
            }
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
        _mcFsChart.timeScale().setVisibleLogicalRange({ from: n - _mcFsVisibleBars, to: n + 12 });

        // Click handler — AVWAP anchor + AVWAP line selection
        _mcFsChart.subscribeClick(function(param) {
            // ── AVWAP anchor ───────────────────────────────────────────────
            if (_mcFsVwapMode) {
                if (!param.time) return;
                var idx = _barIdxByTime(_mcFsOhlcv, param.time);
                if (idx < 0) return;
                _addFsVwap(idx);
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
        leg.style.cssText = 'position:absolute;top:8px;left:14px;z-index:10;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:#8b949e;pointer-events:none;line-height:1.8;background:rgba(13,17,23,0.85);padding:4px 10px;border-radius:4px;';
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
            var cl = d.close >= d.open ? '#089981' : '#b22833';
            var vd = p.seriesData.get(_mcFsVol);
            // Price change from previous candle
            var chgHtml = '';
            var barIdx = _barIdxByTime(_mcFsOhlcv, p.time);
            if (barIdx > 0) {
                var prevClose = _mcFsOhlcv[barIdx - 1].close;
                var delta = d.close - prevClose;
                var pct = (delta / prevClose) * 100;
                var chgClr = delta >= 0 ? '#3fb950' : '#f85149';
                chgHtml = '&nbsp;&nbsp;<span style="color:' + chgClr + '">'
                        + (delta >= 0 ? '+' : '') + delta.toFixed(2)
                        + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)'
                        + '</span>';
            }
            leg.innerHTML =
                '<span style="color:#8b949e">O</span> <span style="color:'+cl+'">'+fp(d.open)+'</span>&nbsp; ' +
                '<span style="color:#8b949e">H</span> <span style="color:'+cl+'">'+fp(d.high)+'</span>&nbsp; ' +
                '<span style="color:#8b949e">L</span> <span style="color:'+cl+'">'+fp(d.low)+'</span>&nbsp; ' +
                '<span style="color:#8b949e">C</span> <span style="color:'+cl+'">'+fp(d.close)+'</span>' +
                chgHtml +
                (vd ? '&nbsp;&nbsp;<span style="color:#6e7681">V</span> <span style="color:#8b949e">'+fv(vd.value)+'</span>' : '');
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

            var chgColor = chg >= 0 ? '#3fb950' : '#f85149';
            var chgSign  = chg >= 0 ? '+' : '';
            var barLabel = _mcFsTf === 'W' ? 'WK' : _mcFsTf === 'M' ? 'MO' : 'DAY';

            // Gradient range bar: red→yellow→green track, dark overlay masks unfilled right,
            // white dot with dark ring marks current price position
            var barColor = chg >= 0 ? '#089981' : '#b22833';

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
                    'border-radius:2px;background:#21262d;vertical-align:middle;flex-shrink:0;overflow:visible;">' +
                    '<span style="position:absolute;left:0;top:0;height:100%;width:' + p + '%;background:' + barColor + ';border-radius:2px;"></span>' +
                    '<span style="position:absolute;top:50%;left:' + p + '%;' +
                    'transform:translate(-50%,-50%);width:8px;height:8px;' +
                    'background:#c9d1d9;border-radius:50%;box-shadow:0 0 0 1.5px #0d1117;"></span>' +
                    crSpan +
                    '</span>';
            }

            // CR% value computed live from day range
            var crRaw   = (dayHigh > dayLow) ? Math.round((close - dayLow) / (dayHigh - dayLow) * 100) : null;
            var crLabel = crRaw != null ? {
                text:  crRaw + '%',
                color: crRaw >= 60 ? '#3fb950' : crRaw >= 30 ? '#e3852b' : '#f85149'
            } : null;

            var adrEl = document.getElementById('mc-fs-mkt-adr');
            var sd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
            if (adrEl) {
                var adrRaw = sd ? sd.adr_pct : null;
                if (adrRaw != null) {
                    adrEl.innerHTML = '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">ADR%</span>'
                                    + '<span style="color:#c9d1d9;font-size:12px;">' + adrRaw.toFixed(1) + '%</span>';
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
                    mcapEl.innerHTML = '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">Mkt Cap</span>'
                                     + '<span style="color:#c9d1d9;font-size:12px;">' + mc + '</span>';
                    mcapEl.style.display = 'inline-flex';
                } else {
                    mcapEl.style.display = 'none';
                }
            }

            document.getElementById('mc-fs-mkt-price').innerHTML =
                '<span style="color:#e6edf3;font-size:20px;font-weight:700;">' + fp(close) + '</span>' +
                '&nbsp;<span style="color:' + chgColor + ';font-size:13px;font-weight:600;">' +
                chgSign + fp(chg) + '&nbsp;(' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)</span>';

            document.getElementById('mc-fs-mkt-day').innerHTML =
                '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">' + barLabel + '</span>' +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(dayLow) + '</span>' +
                mkBar(dayLow, dayHigh, close, 130, crLabel) +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(dayHigh) + '</span>';

            var w52HiPct   = (yrHigh > 0) ? (yrHigh - close) / yrHigh * 100 : 0;
            var w52HiLabel = yrHigh > 0 ? {
                text:  w52HiPct < 0.5 ? 'ATH' : ('-' + w52HiPct.toFixed(1) + '%'),
                color: w52HiPct <= 5 ? '#3fb950' : w52HiPct <= 15 ? '#e3852b' : '#f85149'
            } : null;
            document.getElementById('mc-fs-mkt-52w').innerHTML =
                '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">52W</span>' +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(yrLow) + '</span>' +
                mkBar(yrLow, yrHigh, close, 120, w52HiLabel) +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(yrHigh) + '</span>';

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
            // Alt shortcuts: D = tooltip, T = trendline, A = AVWAP
            if (evt.altKey && !evt.ctrlKey && !evt.metaKey) {
                if (evt.key === 'd' || evt.key === 'D') { evt.preventDefault(); window.mcFsToggleTooltip(); return; }
                if (evt.key === 't' || evt.key === 'T') { evt.preventDefault(); window.mcFsToggleTrendline(); return; }
                if (evt.key === 'a' || evt.key === 'A') { evt.preventDefault(); window.mcFsToggleVwap(); return; }
            }
            if (evt.key !== 'Delete') return;
            // Don't steal Delete from the symbol input
            if (document.getElementById('mc-fs-sym-input')) return;
            // Delete selected trendline first (takes priority over "delete last")
            if (_mcFsSelectedTrendlineIdx !== -1) {
                var selIdx = _mcFsSelectedTrendlineIdx;
                _mcFsSelectedTrendlineIdx = -1;
                var selTl = _mcFsTrendlines.splice(selIdx, 1)[0];
                try { if (_mcFsCandle) _mcFsCandle.detachPrimitive(selTl.primitive); } catch(e) {}
                return;
            }
            // Delete selected AVWAP
            if (_mcFsSelectedVwapIdx !== -1) {
                var selVwapIdx = _mcFsSelectedVwapIdx;
                _mcFsSelectedVwapIdx = -1;
                var removed = _mcFsVwapSeries.splice(selVwapIdx, 1)[0];
                try { _mcFsChart.removeSeries(removed.series); } catch(e) {}
                _mcFsVwapSeries.forEach(function(entry) { entry.series.applyOptions({ lineWidth: 1.5 }); });
                return;
            }
            // Trendline delete (last) when draw tool is active
            if (_mcFsTrendlineMode && _mcFsTrendlines.length) {
                var tLast = _mcFsTrendlines.pop();
                try { if (_mcFsCandle) _mcFsCandle.detachPrimitive(tLast.primitive); } catch(e) {}
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
                ttBtn.title     = 'Data Tooltip';
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

        // Restore trendlines from alert store so they're visible when reviewing the chart
        if (window.alGetTrendlineAlerts) {
            window.alGetTrendlineAlerts(sym).forEach(function(a) {
                _addFsTrendline(a.p1, a.p2);
            });
        }
    }

    // Fullscreen window-level controls
    window.mcFsSetTf = function(tf) {
        if (!_mcFsSym) return;
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
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">Loading\u2026</div>';
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (!document.getElementById('mc-fullscreen-overlay').classList.contains('open')) return;
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
                    (nameStr ? '<span class="mc-fs-dd-name">'  + nameStr + '</span>' : '<span class="mc-fs-dd-name" style="color:#484f58;font-style:italic;">—</span>') +
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
                    (nameStr ? '<span class="mc-fs-dd-name">'  + nameStr + '</span>' : '<span class="mc-fs-dd-name" style="color:#484f58;font-style:italic;">—</span>') +
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
        if (live && live.price && live.prevClose) {
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
                var rkColor  = pctile != null ? (pctile >= 75 ? '#3fb950' : pctile >= 40 ? '#e3852b' : '#f85149') : '#6e7681';
                indRankHtml  = '<span class="mc-cell-hdr-ind-name" style="color:#8b949e;font-size:0.7em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:30%;margin-left:6px;flex-shrink:1;">' + esc(ind) + '</span>'
                             + '<span class="mc-cell-hdr-rank" style="color:' + rkColor + ';">(' + indObj.rank + '/' + total + ')</span>';
            } else if (ind) {
                indRankHtml  = '<span class="mc-cell-hdr-ind-name" style="color:#8b949e;font-size:0.7em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:30%;margin-left:6px;flex-shrink:1;">' + esc(ind) + '</span>';
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
            var chgColor = dayPct > 0 ? '#3fb950' : dayPct < 0 ? '#f85149' : '#484f58';
            var chgStyle = 'color:' + chgColor + ';font-size:0.748em;font-weight:600;flex-shrink:0;font-variant-numeric:tabular-nums;white-space:nowrap;';
            var absDelta = null;
            if (live && live.price && live.prevClose) {
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
                var bg = urgent ? '#3a2008' : '#3a3008';
                var fg = urgent ? '#f0883e' : '#e3c225';
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
                    var mcRankClr = mcPctile != null ? (mcPctile >= 75 ? '#3fb950' : mcPctile >= 40 ? '#e3852b' : '#f85149') : '#6e7681';
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
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">Loading…</div>';
        var openSym = sym;
        // No forced cache-clear here anymore — see the matching note in
        // mcFsSetTf. Reopening a symbol/TF already fetched this session now
        // renders instantly from cache instead of re-entering the queue.
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (!document.getElementById('mc-fullscreen-overlay').classList.contains('open')) return;
            if (_mcFsSym === openSym && _mcFsTf === tf && _mcFsChart) return; // already rendered
            _buildFsChart(sym, ohlcv, tf);
        });
    };

    window.closeMcFullscreen = function() {
        document.getElementById('mc-fullscreen-overlay').classList.remove('open');
        _mcFsDismissCtx();
        if (_mcFsChart) { try { _mcFsChart.remove(); } catch(e) {} _mcFsChart = null; }
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
    function _addWlTrendline(p1, p2) {
        _addTrendlineCore(p1, p2, _wlChart, _wlCandle, _wlOhlcv, _wlTrendlines);
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
            trendlines:   _wlTrendlines,
            contRef:      _wlTrendContRef,
            svgOverlay:   _wlTrendSvgOverlay,
            moveHandler:  _onWlTrendAnchorDragMove,
            endHandler:   _onWlTrendAnchorDragEnd
        });
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
            measurePreviewMoveHandler: _onWlMeasurePreviewMove,
            getSelectedIdx:    function() { return _wlSelectedTrendlineIdx; },
            setSelectedIdx:    function(v) { _wlSelectedTrendlineIdx = v; },
            trendlines:        _wlTrendlines,
            deselectAllTrendlines: _wlDeselectAllTrendlines,
            anchorHitTest:     _wlAnchorHitTest,
            trendlineHitTest:  _wlTrendlineHitTest,
            dragMoveHandler:   _onWlTrendAnchorDragMove,
            dragEndHandler:    _onWlTrendAnchorDragEnd,
            getTrendlineMode:  function() { return _wlTrendlineMode; },
            setTrendlineMode:  function(v) { _wlTrendlineMode = v; },
            getLastCrosshairTime: function() { return _wlLastCrosshairTime; },
            addTrendline:      _addWlTrendline,
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
            trendlineHitTest: _wlTrendlineHitTest
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
            ctxLabelId:     'wl-chart-ctx-label',
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
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;' + (ohlcv === null ? 'cursor:pointer;text-decoration:underline;' : '') + '">' + _wlMsg + '</div>';
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
            _wlTrendSvgLine.setAttribute('stroke', _TRENDLINE_COLOR);
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
            layout: { background: { color: '#0d1117' }, textColor: '#6e7681', panes: { separatorColor: '#161b22', separatorHoverColor: 'rgba(33,38,45,0.5)' } },
            grid:    { vertLines: { visible: false }, horzLines: { visible: false } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: '#21262d', textColor: '#6e7681', scaleMargins: { top: 0.05, bottom: 0.02 } },
            timeScale: { borderColor: '#21262d', timeVisible: false, secondsVisible: false, rightOffset: 12 },
            handleScroll: true, handleScale: true,
        });
        _wlAttachCtxMenu();

        // Candle series
        _wlCandle = _wlChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: '#089981', downColor: '#b22833', borderVisible: false,
            wickUpColor: '#089981', wickDownColor: '#b22833',
            priceLineVisible: false, lastValueVisible: true,
        });
        _wlCandle.setData(_wlOhlcv);

        // Volume pane
        _wlVol = _wlChart.addSeries(LightweightCharts.HistogramSeries, {
            color: '#63a0f8', priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: true,
        }, 1);
        _wlVol.setData(_wlOhlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)' };
        }));
        _wlVol.priceScale().applyOptions({ visible: true, borderColor: '#21262d', textColor: '#6e7681', minimumWidth: 60 });

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
            var color = volDiffPct >= 0 ? '#3fb950' : '#f85149';
            var lbl = document.createElement('div');
            lbl.id = 'wl-chart-vol-pct-label';
            lbl.style.cssText = 'position:absolute;z-index:20;pointer-events:none;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:3px;white-space:nowrap;line-height:1;';
            lbl.innerHTML = '<span style="color:#484f58;">›</span>'
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
                    if (lastX == null || lastX < 0) { lbl.style.display = 'none'; return; }
                    lbl.style.display = 'flex';
                    lbl.style.left = (lastX + 10) + 'px';
                    lbl.style.top  = lblTop;
                }
                positionVolLabel();
                _wlChart.timeScale().subscribeVisibleTimeRangeChange(positionVolLabel);
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
        _wlChart.timeScale().setVisibleLogicalRange({ from: n - _wlVisibleBars, to: n + 12 });

        // Re-render measure overlay on pan/zoom
        _wlChart.timeScale().subscribeVisibleLogicalRangeChange(function() {
            if (_wlMeasureResult) {
                _renderMeasureOverlay(_wlChart, _wlCandle, _wlTrendContRef,
                    _wlMeasureSvgOverlay, _wlMeasureSvgRect, _wlMeasureHLine,
                    _wlMeasureInfoDiv, _wlMeasureResult);
            }
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
                var s = _wlChart.addSeries(LightweightCharts.LineSeries, { color: color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
                s.setData(data);
                _wlVwapSeries.push({ series: s, anchor: idx, color: color, dataMap: dataMap });
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
        leg.style.cssText = 'position:absolute;top:8px;left:14px;z-index:10;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:#8b949e;pointer-events:none;line-height:1.8;background:rgba(13,17,23,0.85);padding:4px 10px;border-radius:4px;';
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
            var cl = d.close >= d.open ? '#089981' : '#b22833';
            var vd = p.seriesData.get(_wlVol);
            var chgHtml = '';
            var barIdx = _barIdxByTime(_wlOhlcv, p.time);
            if (barIdx > 0) {
                var prevClose = _wlOhlcv[barIdx - 1].close;
                var delta = d.close - prevClose;
                var pct = (delta / prevClose) * 100;
                var chgClr = delta >= 0 ? '#3fb950' : '#f85149';
                chgHtml = '&nbsp;&nbsp;<span style="color:' + chgClr + '">'
                        + (delta >= 0 ? '+' : '') + delta.toFixed(2)
                        + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)'
                        + '</span>';
            }
            leg.innerHTML =
                '<span style="color:#8b949e">O</span> <span style="color:'+cl+'">'+fp(d.open)+'</span>&nbsp; ' +
                '<span style="color:#8b949e">H</span> <span style="color:'+cl+'">'+fp(d.high)+'</span>&nbsp; ' +
                '<span style="color:#8b949e">L</span> <span style="color:'+cl+'">'+fp(d.low)+'</span>&nbsp; ' +
                '<span style="color:#8b949e">C</span> <span style="color:'+cl+'">'+fp(d.close)+'</span>' +
                chgHtml +
                (vd ? '&nbsp;&nbsp;<span style="color:#6e7681">Vol</span> <span style="color:#8b949e">' + fv(vd.value) + '</span>' : '');
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
            var chgColor = chg >= 0 ? '#3fb950' : '#f85149';
            var chgSign  = chg >= 0 ? '+' : '';
            var barLabel = tf === 'W' ? 'WK' : tf === 'M' ? 'MO' : 'DAY';
            var barColor = chg >= 0 ? '#089981' : '#b22833';
            function mkBar(low, high, curr, width, crLabel) {
                var pos = (high > low) ? Math.max(2, Math.min(98, (curr - low) / (high - low) * 100)) : 50;
                var p = pos.toFixed(1);
                var crSpan = crLabel != null
                    ? '<span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-150%);font-size:9px;font-weight:700;color:' + crLabel.color + ';letter-spacing:.02em;pointer-events:none;">' + crLabel.text + '</span>'
                    : '';
                return '<span style="position:relative;display:inline-block;width:' + width + 'px;height:4px;border-radius:2px;background:#21262d;vertical-align:middle;flex-shrink:0;overflow:visible;">'
                    + '<span style="position:absolute;left:0;top:0;height:100%;width:' + p + '%;background:' + barColor + ';border-radius:2px;"></span>'
                    + '<span style="position:absolute;top:50%;left:' + p + '%;transform:translate(-50%,-50%);width:8px;height:8px;background:#c9d1d9;border-radius:50%;box-shadow:0 0 0 1.5px #0d1117;"></span>'
                    + crSpan + '</span>';
            }
            var crRaw   = (dayHigh > dayLow) ? Math.round((close - dayLow) / (dayHigh - dayLow) * 100) : null;
            var crLabel = crRaw != null ? { text: crRaw + '%', color: crRaw >= 60 ? '#3fb950' : crRaw >= 30 ? '#e3852b' : '#f85149' } : null;
            var adrEl = document.getElementById('wl-chart-mkt-adr');
            if (adrEl) {
                var adrSd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var adrRaw = adrSd ? adrSd.adr_pct : null;
                if (adrRaw != null) {
                    adrEl.innerHTML = '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">ADR%</span>'
                                    + '<span style="color:#c9d1d9;font-size:12px;">' + adrRaw.toFixed(1) + '%</span>';
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
                    mcapEl.innerHTML = '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">Mkt Cap</span><span style="color:#c9d1d9;font-size:12px;">' + mc + '</span>';
                    mcapEl.style.display = 'inline-flex';
                } else { mcapEl.style.display = 'none'; }
            }
            document.getElementById('wl-chart-mkt-price').innerHTML =
                '<span style="color:#e6edf3;font-size:20px;font-weight:700;">' + fp(close) + '</span>' +
                '&nbsp;<span style="color:' + chgColor + ';font-size:13px;font-weight:600;">' + chgSign + fp(chg) + '&nbsp;(' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)</span>';
            document.getElementById('wl-chart-mkt-day').innerHTML =
                '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">' + barLabel + '</span>' +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(dayLow) + '</span>' +
                mkBar(dayLow, dayHigh, close, 130, crLabel) +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(dayHigh) + '</span>';
            var w52HiPct   = (yrHigh > 0) ? (yrHigh - close) / yrHigh * 100 : 0;
            var w52HiLabel = yrHigh > 0 ? {
                text:  w52HiPct < 0.5 ? 'ATH' : ('-' + w52HiPct.toFixed(1) + '%'),
                color: w52HiPct <= 5 ? '#3fb950' : w52HiPct <= 15 ? '#e3852b' : '#f85149'
            } : null;
            document.getElementById('wl-chart-mkt-52w').innerHTML =
                '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">52W</span>' +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(yrLow) + '</span>' +
                mkBar(yrLow, yrHigh, close, 120, w52HiLabel) +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(yrHigh) + '</span>';
            document.getElementById('wl-chart-mkt-info').style.display = 'flex';
        })();

        // Keyboard: Delete/Escape for trendlines + AVWAP
        if (_wlKeyHandler) { document.removeEventListener('keydown', _wlKeyHandler); }
        _wlKeyHandler = function(evt) {
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
            if (_wlSelectedTrendlineIdx !== -1) {
                var selIdx = _wlSelectedTrendlineIdx;
                _wlSelectedTrendlineIdx = -1;
                var selTl = _wlTrendlines.splice(selIdx, 1)[0];
                try { if (_wlCandle) _wlCandle.detachPrimitive(selTl.primitive); } catch(e) {}
                return;
            }
            if (_wlSelectedVwapIdx !== -1) {
                var selVwapIdx = _wlSelectedVwapIdx;
                _wlSelectedVwapIdx = -1;
                var removed = _wlVwapSeries.splice(selVwapIdx, 1)[0];
                try { _wlChart.removeSeries(removed.series); } catch(e) {}
                _wlVwapSeries.forEach(function(entry) { entry.series.applyOptions({ lineWidth: 1.5 }); });
                return;
            }
            if (_wlTrendlineMode && _wlTrendlines.length) {
                var tLast = _wlTrendlines.pop();
                try { if (_wlCandle) _wlCandle.detachPrimitive(tLast.primitive); } catch(e) {}
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
                ttBtn.title     = 'Data Tooltip';
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

        // Restore trendlines from alert store so they're visible when reviewing the chart
        if (window.alGetTrendlineAlerts) {
            window.alGetTrendlineAlerts(sym).forEach(function(a) {
                _addWlTrendline(a.p1, a.p2);
            });
        }
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
        delete _mcOhlcvCache[_wlSym + '_' + tf];
        var sym = _wlSym;
        var container = document.getElementById('wl-chart-widget');
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">Loading\u2026</div>';
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (_wlSym !== sym) return;
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

