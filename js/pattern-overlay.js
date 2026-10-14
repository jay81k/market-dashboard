// ── Pattern overlay: base / cup / double bottom on the fullscreen chart ──
// Phase 2: draws the output of pattern.js (PatternDetector) as a Lightweight
// Charts v5 series primitive. Fullscreen chart only, daily timeframe only
// (like the Pine script). Off by default; the toolbar button
// (#mc-fs-pattern-btn) toggles it and remembers the choice.
//
// Wiring:
//   - index.html loads pattern.js and this file before multichart.js
//   - _buildFsChart() in multichart.js calls PatternOverlay.attach(...) once
//     the chart and candle series exist (one primitive per chart build; the
//     primitive goes away with the chart, so there is nothing to clean up)
//   - the detector only re-runs when the last candle changes, so the
//     10-second live tick keeps the current pattern up to date
//
// Colours and line weights follow the Pine script (green lines, blue / red /
// green trade boxes). They are fixed on purpose so the chart matches
// TradingView in both themes.
//
// Debug: PatternOverlay.last() returns the latest detector result
// (events, lines, boxes, cups, state) for the chart currently on screen.
(function() {
    var STORAGE_KEY = 'dashboard-patterns-on';
    var LINE_COLOR  = 'rgb(146, 193, 131)';
    var BOX_FILL = {
        entry:      'rgba(0, 0, 255, 0.08)',
        stopLoss:   'rgba(255, 0, 0, 0.08)',
        takeProfit: 'rgba(60, 243, 4, 0.12)'
    };

    var enabled = false;
    try { enabled = localStorage.getItem(STORAGE_KEY) === '1'; } catch (e) {}
    var current = null;   // { series, primitive, requestUpdate, lastResult } for the chart on screen

    function syncButton() {
        var b = document.getElementById('mc-fs-pattern-btn');
        if (b) b.classList.toggle('active', enabled);
    }

    function detach() {
        if (!current) return;
        try { current.series.detachPrimitive(current.primitive); } catch (e) {}
        current = null;
    }

    // chart/series: the fullscreen chart and its candle series
    // getOhlcv():   the live candle array (mutated in place by the live tick)
    // getTf():      current timeframe ('D' | 'W' | 'M' ...)
    function attach(chart, series, getOhlcv, getTf) {
        detach();
        var handle = { series: series, primitive: null, requestUpdate: null, lastResult: null };
        var cacheKey = '';

        function compute() {
            var o = getOhlcv();
            if (!o || o.length < 40 || getTf() !== 'D' || !window.PatternDetector) return null;
            var last = o[o.length - 1];
            var key = o.length + '|' + last.time + '|' + last.high + '|' + last.low + '|' + last.close;
            if (key !== cacheKey) {
                cacheKey = key;
                try { handle.lastResult = window.PatternDetector.detect(o, { timeframe: 'D' }); }
                catch (e) { handle.lastResult = null; if (window.console) console.warn('PatternDetector failed', e); }
            }
            return handle.lastResult;
        }

        function xOf(o, idx) { var b = o[idx]; return b ? chart.timeScale().timeToCoordinate(b.time) : null; }
        function yOf(p) { return series.priceToCoordinate(p); }

        // Trade boxes sit behind the candles
        var boxesView = {
            zOrder: function() { return 'bottom'; },
            renderer: function() {
                return {
                    draw: function(target) {
                        var r = compute();
                        if (!r || !r.boxes.length) return;
                        var o = getOhlcv();
                        target.useBitmapCoordinateSpace(function(scope) {
                            var ctx = scope.context, rx = scope.horizontalPixelRatio, ry = scope.verticalPixelRatio;
                            r.boxes.forEach(function(b) {
                                var x1 = xOf(o, b.idx1), x2 = xOf(o, b.idx2), y1 = yOf(b.top), y2 = yOf(b.bottom);
                                if (x1 == null || x2 == null || y1 == null || y2 == null) return;
                                ctx.fillStyle = BOX_FILL[b.kind] || BOX_FILL.entry;
                                ctx.fillRect(Math.min(x1, x2) * rx, Math.min(y1, y2) * ry,
                                             Math.max(Math.abs(x2 - x1), 1) * rx, Math.abs(y2 - y1) * ry);
                            });
                        });
                    }
                };
            }
        };

        // Base lines, double-bottom legs, cup curves and cup handles sit on top of the candles
        var linesView = {
            zOrder: function() { return 'normal'; },
            renderer: function() {
                return {
                    draw: function(target) {
                        var r = compute();
                        if (!r || (!r.lines.length && !r.cups.length)) return;
                        var o = getOhlcv();
                        target.useBitmapCoordinateSpace(function(scope) {
                            var ctx = scope.context, rx = scope.horizontalPixelRatio, ry = scope.verticalPixelRatio;
                            ctx.save();
                            ctx.strokeStyle = LINE_COLOR;
                            ctx.lineJoin = 'round';
                            r.lines.forEach(function(l) {
                                var x1 = xOf(o, l.idx1), x2 = xOf(o, l.idx2), y1 = yOf(l.p1), y2 = yOf(l.p2);
                                if (x1 == null || x2 == null || y1 == null || y2 == null) return;
                                ctx.beginPath();
                                ctx.moveTo(x1 * rx, y1 * ry);
                                ctx.lineTo(x2 * rx, y2 * ry);
                                ctx.lineWidth = l.width * rx;
                                if (l.dotted) { ctx.lineCap = 'round'; ctx.setLineDash([0.1 * rx, l.width * 2 * rx]); }
                                else          { ctx.lineCap = 'butt';  ctx.setLineDash([]); }
                                ctx.stroke();
                            });
                            ctx.setLineDash([]);
                            ctx.lineCap = 'round';
                            function strokeCurve(pts) {
                                if (!pts || pts.length < 2) return;
                                ctx.beginPath();
                                ctx.lineWidth = 3 * rx;
                                var pen = false;
                                pts.forEach(function(p) {
                                    var x = xOf(o, p.idx), y = yOf(p.price);
                                    if (x == null || y == null) { pen = false; return; }
                                    if (!pen) { ctx.moveTo(x * rx, y * ry); pen = true; }
                                    else ctx.lineTo(x * rx, y * ry);
                                });
                                ctx.stroke();
                            }
                            r.cups.forEach(function(c) {
                                strokeCurve(window.PatternDetector.cupCurve(c));
                                // handle after the cup's right side (empty when the cup has none)
                                if (window.PatternDetector.handleCurve) strokeCurve(window.PatternDetector.handleCurve(c));
                            });
                            ctx.restore();
                        });
                    }
                };
            }
        };

        var primitive = {
            attached: function(param) { handle.requestUpdate = function() { try { param.requestUpdate(); } catch (e) {} }; },
            detached: function() { handle.requestUpdate = null; },
            paneViews: function() { return enabled ? [boxesView, linesView] : []; }
        };

        handle.primitive = primitive;
        series.attachPrimitive(primitive);
        current = handle;
    }

    window.mcFsTogglePatterns = function() {
        enabled = !enabled;
        try { localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0'); } catch (e) {}
        syncButton();
        if (current && current.requestUpdate) current.requestUpdate();
    };

    window.PatternOverlay = {
        attach: attach,
        detach: detach,
        isEnabled: function() { return enabled; },
        last: function() { return current ? current.lastResult : null; }
    };

    syncButton();
    document.addEventListener('DOMContentLoaded', syncButton);
})();
