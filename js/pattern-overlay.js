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
//   - the detector also re-runs when the chart's price scale mode changes
//     (linear <-> log), because the double-bottom W is placed for that scale
//
// Colours and line weights follow the Pine script (green lines, blue / red /
// green trade boxes). They are fixed on purpose so the chart matches
// TradingView in both themes.
//
// Hover tooltip (only while the patterns toggle is on):
//   - moving the cursor within 8px of a drawn pattern line / cup curve shows a
//     card with the pattern's name, length in trading days, depth, price range
//     and status (Forming / Broke out); the pattern also gets a soft highlight
//   - what is shown comes from result.patterns (pattern.js); this file only
//     hit-tests the same geometry it draws, so the two can never disagree
//   - the card is a child of the chart's own element (chart.chartElement()), so
//     no other file needs to know about it; it is removed with the chart
//
// Debug: PatternOverlay.last() returns the latest detector result
// (events, lines, boxes, cups, patterns, state) for the chart currently on screen;
// PatternOverlay.hovered() returns the id of the pattern under the cursor.
(function() {
    var STORAGE_KEY = 'dashboard-patterns-on';
    var LINE_COLOR  = 'rgb(146, 193, 131)';
    var BOX_FILL = {
        entry:      'rgba(0, 0, 255, 0.08)',
        stopLoss:   'rgba(255, 0, 0, 0.08)',
        takeProfit: 'rgba(60, 243, 4, 0.12)'
    };

    // Hover tooltip
    var HIT_TOLERANCE = 8;                          // px: how close the cursor must be to a drawn line / curve
    var HOVER_FILL    = 'rgba(146, 193, 131, 0.09)';
    var HOVER_GLOW    = 'rgba(146, 193, 131, 0.30)';
    var TYPE_COLOR    = { flatBase: '#8fc27f', deepBase: '#d9a441', cup: '#5aa6e0', cupHandle: '#5aa6e0', doubleBottom: '#b48af2' };
    var STATUS_TEXT   = { forming: 'Forming', brokeOut: 'Broke out' };
    var CSS_ID        = 'mc-pattern-tip-style';
    var CSS = [
        '.mc-pattern-tip{position:absolute;left:0;top:0;z-index:30;pointer-events:none;box-sizing:border-box;',
        'min-width:178px;padding:9px 12px 10px 11px;border-radius:7px;border:1px solid rgba(255,255,255,.08);',
        'border-left:3px solid var(--mcpt-accent,#8fc27f);background:rgba(22,28,39,.97);color:#e8ebf1;',
        'font:12px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;',
        'font-variant-numeric:tabular-nums;white-space:nowrap;box-shadow:0 8px 22px rgba(0,0,0,.38);',
        'opacity:0;visibility:hidden;transition:opacity .12s ease-out}',
        '.mc-pattern-tip.mcpt-on{opacity:1;visibility:visible}',
        '.mcpt-head{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:6px}',
        '.mcpt-name{font-size:13px;font-weight:600}',
        '.mcpt-chip{font-size:11px;line-height:1;padding:3px 7px;border-radius:999px;border:1px solid transparent}',
        '.mcpt-chip.forming{color:#e6b95c;border-color:rgba(230,185,92,.45)}',
        '.mcpt-chip.brokeOut{color:#9bd58b;background:rgba(155,213,139,.14)}',
        '.mcpt-row{display:flex;justify-content:space-between;gap:18px;padding:1.5px 0}',
        '.mcpt-k{color:#8b94a5}',
        '.mcpt-v{color:#e8ebf1;text-align:right}',
        '@media (prefers-reduced-motion:reduce){.mc-pattern-tip{transition:none}}'
    ].join('');

    function injectCss() {
        if (document.getElementById(CSS_ID)) return;
        var st = document.createElement('style');
        st.id = CSS_ID;
        st.textContent = CSS;
        (document.head || document.documentElement).appendChild(st);
    }

    // shortest distance from (px, py) to the segment a-b
    function distToSeg(px, py, a, b) {
        var dx = b.x - a.x, dy = b.y - a.y, len2 = dx * dx + dy * dy;
        var t = len2 ? ((px - a.x) * dx + (py - a.y) * dy) / len2 : 0;
        t = Math.max(0, Math.min(1, t));
        var ex = px - (a.x + t * dx), ey = py - (a.y + t * dy);
        return Math.sqrt(ex * ex + ey * ey);
    }

    var enabled = false;
    try { enabled = localStorage.getItem(STORAGE_KEY) === '1'; } catch (e) {}
    var current = null;   // { series, primitive, requestUpdate, lastResult } for the chart on screen

    function syncButton() {
        var b = document.getElementById('mc-fs-pattern-btn');
        if (b) b.classList.toggle('active', enabled);
    }

    function detach() {
        if (!current) return;
        if (current.cleanup) current.cleanup();
        try { current.series.detachPrimitive(current.primitive); } catch (e) {}
        current = null;
    }

    // chart/series: the fullscreen chart and its candle series
    // getOhlcv():   the live candle array (mutated in place by the live tick)
    // getTf():      current timeframe ('D' | 'W' | 'M' ...)
    function attach(chart, series, getOhlcv, getTf) {
        detach();
        var handle = { series: series, primitive: null, requestUpdate: null, lastResult: null,
                       hoverId: null, hoverKey: '', cleanup: null, clearHover: null };
        var cacheKey = '';

        // The double-bottom W is placed below the candles for the scale the chart
        // is drawn on, so the detector has to know it. 'log' for a logarithmic
        // price scale; percentage / indexed-to-100 are linear in price. The mode
        // is part of the cache key, so switching scale re-places the W.
        function scaleMode() {
            try { return series.priceScale().options().mode === 1 ? 'log' : 'linear'; }
            catch (e) { return 'linear'; }
        }

        function compute() {
            var o = getOhlcv();
            if (!o || o.length < 40 || getTf() !== 'D' || !window.PatternDetector) return null;
            var last = o[o.length - 1];
            var mode = scaleMode();
            var key = mode + '|' + o.length + '|' + last.time + '|' + last.high + '|' + last.low + '|' + last.close;
            if (key !== cacheKey) {
                cacheKey = key;
                try { handle.lastResult = window.PatternDetector.detect(o, { timeframe: 'D', priceScale: mode }); }
                catch (e) { handle.lastResult = null; if (window.console) console.warn('PatternDetector failed', e); }
            }
            return handle.lastResult;
        }

        function xOf(o, idx) { var b = o[idx]; return b ? chart.timeScale().timeToCoordinate(b.time) : null; }
        function yOf(p) { return series.priceToCoordinate(p); }

        // ── hover: geometry, hit test, highlight, card ───────────────────
        // The hit test uses the very same coordinates the drawing uses (xOf / yOf
        // on result.lines and the cup curves), so what you can hover is exactly
        // what you can see.

        // every drawn path of a pattern as [{x, y}, ...] in chart pixels
        function project(p, r, o) {
            var paths = [];
            (p.lineIdx || []).forEach(function(i) {
                var l = r.lines[i];
                if (!l) return;
                var x1 = xOf(o, l.idx1), x2 = xOf(o, l.idx2), y1 = yOf(l.p1), y2 = yOf(l.p2);
                if (x1 == null || x2 == null || y1 == null || y2 == null) return;
                paths.push([{ x: x1, y: y1 }, { x: x2, y: y2 }]);
            });
            var c = (p.cupIdx != null && p.cupIdx >= 0) ? r.cups[p.cupIdx] : null;
            if (c && window.PatternDetector) {
                [window.PatternDetector.cupCurve(c),
                 window.PatternDetector.handleCurve ? window.PatternDetector.handleCurve(c) : null].forEach(function(pts) {
                    if (!pts || pts.length < 2) return;
                    var path = [];
                    pts.forEach(function(q) {
                        var x = xOf(o, q.idx), y = yOf(q.price);
                        if (x != null && y != null) path.push({ x: x, y: y });
                    });
                    if (path.length > 1) paths.push(path);
                });
            }
            return paths;
        }

        function hitTest(r, o, px, py) {
            var best = null, bestD = HIT_TOLERANCE + 1;
            r.patterns.forEach(function(p) {
                var xa = xOf(o, p.startIdx), xb = xOf(o, p.endIdx);
                if (xa != null && xb != null &&
                    (px < Math.min(xa, xb) - HIT_TOLERANCE || px > Math.max(xa, xb) + HIT_TOLERANCE)) return;
                var d = Infinity;
                project(p, r, o).forEach(function(path) {
                    for (var i = 1; i < path.length; i++) {
                        var s = distToSeg(px, py, path[i - 1], path[i]);
                        if (s < d) d = s;
                    }
                });
                if (d > HIT_TOLERANCE) return;
                // closest line wins; on a tie (within half a pixel) the shorter pattern wins
                if (d < bestD - 0.5 || (Math.abs(d - bestD) <= 0.5 && best && p.bars < best.bars)) { best = p; bestD = d; }
            });
            return best;
        }

        function hoveredPattern() {
            if (!handle.hoverId) return null;
            var r = compute();
            if (!r || !r.patterns) return null;
            for (var i = 0; i < r.patterns.length; i++) {
                if (r.patterns[i].id === handle.hoverId) return { p: r.patterns[i], r: r };
            }
            return null;
        }

        var host = null, tip = null, dragging = false, cleaned = false;
        try { host = chart.chartElement ? chart.chartElement() : null; } catch (e) { host = null; }

        function fmtPrice(v) {
            try { return series.priceFormatter().format(v); } catch (e) { return v.toFixed(2); }
        }
        function el(tag, cls, text) {
            var n = document.createElement(tag);
            if (cls) n.className = cls;
            if (text != null) n.textContent = text;
            return n;
        }

        function fillTip(p) {
            while (tip.firstChild) tip.removeChild(tip.firstChild);
            tip.style.setProperty('--mcpt-accent', TYPE_COLOR[p.type] || LINE_COLOR);
            var head = el('div', 'mcpt-head');
            head.appendChild(el('span', 'mcpt-name', p.label));
            if (STATUS_TEXT[p.status]) head.appendChild(el('span', 'mcpt-chip ' + p.status, STATUS_TEXT[p.status]));
            tip.appendChild(head);
            function row(k, v) {
                var rw = el('div', 'mcpt-row');
                rw.appendChild(el('span', 'mcpt-k', k));
                rw.appendChild(el('span', 'mcpt-v', v));
                tip.appendChild(rw);
            }
            var len = p.bars + (p.bars === 1 ? ' day' : ' days');
            if (p.weeks >= 1) len += ' (' + p.weeks + (p.weeks === 1 ? ' wk)' : ' wks)');
            row('Length', len);
            row('Depth', p.depthPct.toFixed(1) + '%');
            row('Range', fmtPrice(p.top) + ' \u2013 ' + fmtPrice(p.bottom));
            if (p.type === 'cupHandle' && p.handleBars != null) {
                row('Handle', p.handleBars + (p.handleBars === 1 ? ' day, ' : ' days, ') + p.handlePullbackPct.toFixed(1) + '% pullback');
            }
            if (p.type === 'doubleBottom' && p.undercutPct != null) {
                row('Second low', p.undercutPct.toFixed(1) + '% below first');
            }
        }

        function placeTip(x, y) {
            if (!tip || !host) return;
            var w = tip.offsetWidth, h = tip.offsetHeight, W = host.clientWidth, H = host.clientHeight;
            var tx = x + 16, ty = y + 18;
            if (tx + w > W - 4) tx = x - w - 16;
            if (ty + h > H - 4) ty = y - h - 14;
            tx = Math.max(4, tx); ty = Math.max(4, ty);
            tip.style.transform = 'translate(' + Math.round(tx) + 'px,' + Math.round(ty) + 'px)';
        }

        function hideTip() { if (tip) tip.classList.remove('mcpt-on'); }

        function clearHover() {
            hideTip();
            if (handle.hoverId !== null) {
                handle.hoverId = null; handle.hoverKey = '';
                if (handle.requestUpdate) handle.requestUpdate();
            }
        }
        handle.clearHover = clearHover;

        function onMove(param) {
            if (!enabled || dragging || !param || !param.point) { clearHover(); return; }
            if (param.paneIndex !== undefined) {
                var ok = true;
                try { ok = series.getPane().paneIndex() === param.paneIndex; } catch (e) { ok = param.paneIndex === 0; }
                if (!ok) { clearHover(); return; }
            }
            var r = compute();
            if (!r || !r.patterns || !r.patterns.length) { clearHover(); return; }
            var hit = hitTest(r, getOhlcv(), param.point.x, param.point.y);
            if (!hit) { clearHover(); return; }
            if (!tip && host) {
                injectCss();
                tip = document.createElement('div');
                tip.className = 'mc-pattern-tip';
                host.appendChild(tip);
            }
            var key = hit.id + '|' + hit.endIdx + '|' + hit.status + '|' + hit.bars;
            if (handle.hoverId !== hit.id) {
                handle.hoverId = hit.id;
                if (handle.requestUpdate) handle.requestUpdate();
            }
            if (tip && handle.hoverKey !== key) { handle.hoverKey = key; fillTip(hit); }
            if (tip) { tip.classList.add('mcpt-on'); placeTip(param.point.x, param.point.y); }
        }

        // zooming / scrolling moves the pattern from under a still cursor: drop the card until the next mouse move
        function onRange() { clearHover(); }
        function onDown() { dragging = true; clearHover(); }
        function onUp() { dragging = false; }

        try { chart.subscribeCrosshairMove(onMove); } catch (e) {}
        try { chart.timeScale().subscribeVisibleLogicalRangeChange(onRange); } catch (e) {}
        if (host) host.addEventListener('mousedown', onDown);
        window.addEventListener('mouseup', onUp);

        handle.cleanup = function() {
            if (cleaned) return;
            cleaned = true;
            try { chart.unsubscribeCrosshairMove(onMove); } catch (e) {}
            try { chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange); } catch (e) {}
            if (host) host.removeEventListener('mousedown', onDown);
            window.removeEventListener('mouseup', onUp);
            if (tip && tip.parentNode) tip.parentNode.removeChild(tip);
            tip = null;
            handle.hoverId = null; handle.hoverKey = '';
        };

        // bounding box of a pattern's drawn paths
        function boundsOf(paths) {
            var b = { x1: Infinity, y1: Infinity, x2: -Infinity, y2: -Infinity };
            paths.forEach(function(path) {
                path.forEach(function(q) {
                    if (q.x < b.x1) b.x1 = q.x; if (q.x > b.x2) b.x2 = q.x;
                    if (q.y < b.y1) b.y1 = q.y; if (q.y > b.y2) b.y2 = q.y;
                });
            });
            return b.x1 <= b.x2 ? b : null;
        }

        // soft tint over the hovered pattern, behind the candles
        var hoverFillView = {
            zOrder: function() { return 'bottom'; },
            renderer: function() {
                return {
                    draw: function(target) {
                        var h = hoveredPattern();
                        if (!h) return;
                        var b = boundsOf(project(h.p, h.r, getOhlcv()));
                        if (!b) return;
                        target.useBitmapCoordinateSpace(function(scope) {
                            var ctx = scope.context, rx = scope.horizontalPixelRatio, ry = scope.verticalPixelRatio;
                            ctx.fillStyle = HOVER_FILL;
                            ctx.fillRect((b.x1 - 4) * rx, (b.y1 - 6) * ry, (b.x2 - b.x1 + 8) * rx, (b.y2 - b.y1 + 12) * ry);
                        });
                    }
                };
            }
        };

        // soft glow along the hovered pattern's lines, under the real lines
        var hoverGlowView = {
            zOrder: function() { return 'normal'; },
            renderer: function() {
                return {
                    draw: function(target) {
                        var h = hoveredPattern();
                        if (!h) return;
                        var paths = project(h.p, h.r, getOhlcv());
                        if (!paths.length) return;
                        target.useBitmapCoordinateSpace(function(scope) {
                            var ctx = scope.context, rx = scope.horizontalPixelRatio, ry = scope.verticalPixelRatio;
                            ctx.save();
                            ctx.strokeStyle = HOVER_GLOW;
                            ctx.lineWidth = 7 * rx;
                            ctx.lineCap = 'round';
                            ctx.lineJoin = 'round';
                            ctx.setLineDash([]);
                            paths.forEach(function(path) {
                                ctx.beginPath();
                                path.forEach(function(q, i) {
                                    if (i === 0) ctx.moveTo(q.x * rx, q.y * ry); else ctx.lineTo(q.x * rx, q.y * ry);
                                });
                                ctx.stroke();
                            });
                            ctx.restore();
                        });
                    }
                };
            }
        };

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
            detached: function() { handle.requestUpdate = null; if (handle.cleanup) handle.cleanup(); },
            paneViews: function() { return enabled ? [boxesView, hoverFillView, hoverGlowView, linesView] : []; }
        };

        handle.primitive = primitive;
        series.attachPrimitive(primitive);
        current = handle;
    }

    window.mcFsTogglePatterns = function() {
        enabled = !enabled;
        try { localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0'); } catch (e) {}
        syncButton();
        if (!enabled && current && current.clearHover) current.clearHover();
        if (current && current.requestUpdate) current.requestUpdate();
    };

    window.PatternOverlay = {
        attach: attach,
        detach: detach,
        isEnabled: function() { return enabled; },
        last: function() { return current ? current.lastResult : null; },
        hovered: function() { return current ? current.hoverId : null; }
    };

    syncButton();
    document.addEventListener('DOMContentLoaded', syncButton);
})();
