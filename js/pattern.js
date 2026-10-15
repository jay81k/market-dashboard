// ── Chart pattern detector: base / cup / double bottom ───────────────────
// Faithful port of the TradingView Pine "Chart Pattern Recognition" script.
// Pure logic: no DOM, no chart library, no network. Feed it daily candles,
// get back what the Pine script would have drawn plus an event log.
//
//   var r = PatternDetector.detect(candles, opts);
//   candles: [{time, open, high, low, close, volume}, ...] sorted ascending
//
// PHASE 1 SCOPE: this is a PORT, bugs included, except for the deliberate
// deviations listed under DEVIATIONS FROM PINE below.
// Where Pine behaviour was ambiguous it is listed under ASSUMPTIONS and must
// be checked against TradingView before the output is trusted.
//
// ── DEVIATIONS FROM PINE (deliberate) ────────────────────────────────────
//  - DETECTION: a double bottom needs at least opts.dbMinLengthWeeks weeks
//    (default 7 = 35 bars) between the old high (first pivot high of the W)
//    and the bar where the W is detected. Pine had no minimum, only the
//    dbLengthWeeks maximum. Same measure as the cup's 30-bar minimum
//    (n - startBaseBar); detection is delayed until the base is old enough,
//    and a W whose timing windows close before that is never detected.
//    Set dbMinLengthWeeks: 0 to get Pine's behaviour back.
//  - DRAWING ONLY: the double-bottom leg lines ('dbLeg1'/'dbLeg2'/'dbLeg3') take their y
//    from the low of the pivot-high candle itself (x 0.99), looked up by
//    pivotHighBars. Pine read them from lowOfpivotHighPrices, which is fed by
//    pivot highs AND base starts, so its entries drift away from the pivots
//    and the middle peak of the W could be drawn at the old high's level,
//    above its own candles. That array is not ported. Detection is unchanged:
//    it was only ever read when these three lines were created.
//  - DOUBLE BOTTOM LEGS (drawing only): the W ('dbLeg1'..'dbLeg3' + 'dbLow') is
//    placed again so that no leg runs through a candle: each of its five
//    vertices sits as high as the others allow, below every candle its legs
//    span. Each vertex starts on the clear level of its own candle and can
//    only move down; every candle under a leg pushes the two vertices of that
//    leg down far enough to clear it; every leg keeps at least
//    opts.dbLegKeepPct (50) of its original height so the W stays a W; and of
//    all placements that do this, the one with the smallest total downward
//    movement is used (an exact linear programme, see minSumShifts()).
//    The vertices are therefore NOT on the pivots (they sit at or below them),
//    so read pivot prices off the candles, not off the W. The dotted 'dbHigh'
//    buy line is not moved.
//    opts.dbLegGapPct (0.3) is the gap kept to the candles, opts.dbLegClear
//    ('wick' | 'body') says whether wick tips count, opts.priceScale ('linear'
//    | 'log') must match the chart's scale for the legs to clear in both.
//    A straight W that sits ON its pivots and also stays below the candles
//    does not exist: the candles next to a trough sit just above it, which
//    flattens any straight leg that clears them. On a clean W this lands within
//    a few percent of the pivots; on a messy one (a steep drop right after the
//    middle peak) it can be much further, and dbLegKeepPct trades that against
//    how much of a W is left. If the solver ever fails the W is moved down as
//    one rigid piece, which always clears. Recomputed on every call, so it can
//    move when a new bar changes the tightest candle.
//  - CUP HANDLES (drawing only): Pine never draws a handle. When a cup is
//    recorded (live on the last bar, or on the breakout bar), cupRecord() also
//    looks for a handle and stores it as cup.handle (null when there is none;
//    cup.handleReject then says which rule rejected it, for tuning).
//    Cup detection itself is NOT changed: a cup that fails the handle tests is
//    still recorded and drawn exactly as before, just without a handle.
//    A handle is only accepted when ALL of these hold (opts.handle*):
//      cup:    at least 7 weeks (35 bars) from the old high to now, depth
//              12%..45%, and at least a 30% prior uptrend, measured as the old
//              high vs the lowest low of the 126 bars up to and including it
//              (fewer if the data is shorter). Pine's legUpCond looks at a
//              103-bar window that includes the base itself, so it is NOT a
//              real prior-uptrend test and is left alone because it also
//              drives base detection
//      lip:    a running peak of the highs after the cup low; it must be above
//              the middle of the cup and within 15% of the old high
//      handle: the bars after the lip, up to the bar before price takes out
//              the lip high (the pivot), or up to the latest bar if it has not
//              been taken out yet; at least 5 bars; its lowest low is at least
//              3% below the lip high and stays above the middle of the cup.
//              If several candidates qualify, the latest one is used.
//              Volume is ignored on purpose.
//    With a handle the cup's right side ends at the lip (Pine ended it at the
//    low of the bar before the last one, which swallowed the handle).
//    The handle itself is drawn as one straight line from the lip candle's low
//    down to the handle's lowest low (see handleCurve()).
//    BUY POINT: a cup with a handle is bought through the handle high (= the
//    lip high, the price that has to be taken out), not through the old high.
//    For such a cup:
//      line:   the dotted line ('handleHigh') sits on the handle high, from the
//              lip bar to the bar that takes it out (or to the latest bar while
//              the handle is still open). The old-high 'baseHigh' line is dropped
//      event:  the 'breakout' event moves to the bar that takes the lip high out
//              (extra field handle: true); the old-high one is removed
//      boxes:  entry / stop / target boxes start on that bar, anchored to the lip
//              high (same 1.05 / 0.95-0.92 / 1.20-1.25 levels and the same
//              extension rule as Pine). They are only drawn where the old-high
//              boxes would still have been on the chart, or for a live cup
//    Cups without a handle keep the old-high line, event and boxes.
//    NOT changed (still describe the old-high version): res.state (baseTop,
//    inTradeBox) and the base lifecycle itself. A cup is still only recorded
//    when price takes out the old high or while it is live on the latest bar, so
//    a handle breakout that fails before reaching the old high is not drawn.
//    Not checkable here: the "no more than 2.5x the market's correction" rule
//    (the detector only sees one stock). Set handles: false for Pine's output.
//
// ── ASSUMPTIONS (UNVERIFIED against TradingView) ─────────────────────────
//  1. ta.pivothigh/low tie rule (equal highs/lows inside the window).
//     Controlled by opts.pivotTie: 'both' | 'earlier' | 'later'.
//  2. Pine "for" loops with fractional bounds (the cup tests use
//     baseCount/3 and baseCount/4): counter runs inclusive, direction taken
//     from the bounds, series offset = floor(counter). May differ from
//     TradingView by one bar at a loop edge.
//  3. line.delete()/box.delete() only hide a drawing. Deleted line objects
//     keep their coordinates and later logic still reads them, as Pine does.
//  4. Division by zero in condPricesF is treated as "condition false".
//
// ── PINE QUIRKS REPRODUCED ON PURPOSE ────────────────────────────────────
//  - condFourthTwo is never assigned, so the "fourth" cup branch is dead.
//  - condThirdTwo is assigned twice; only the second (quarter based) counts.
//  - The double-bottom "properties" check has an operator-precedence quirk:
//    (isDB and A) or B.
//  - Take-profit box is built with top < bottom (20% .. 25% above entry).
//
// ── NOT PORTED (no effect on detection) ──────────────────────────────────
//  baseCloses (written, never read), isCup (sticky, never read), topDBPrice
//  (never read), lowCompare (only used in commented code), first condThirdTwo
//  loop (overwritten), line/array memory trimming (drawing only), Pine's
//  max_lines_count eviction (the renderer decides how much to show).
//
// Daily timeframe only, like the Pine script. Other timeframes return
// { supported: false }.
(function() {
    'use strict';

    var DEFAULTS = {
        timeframe: 'D',
        pivot: 9,                  // i_pivot
        baseDepthPct: 50,          // i_baseDepth (%)
        baseLengthWeeks: 65,       // i_baseLength (weeks, x5 = trading days)
        dbDepthPct: 50,            // i_dbDepth (%)
        dbLengthWeeks: 65,         // i_dbLength (weeks, x5 = trading days)
        dbMinLengthWeeks: 7,       // NOT in Pine: minimum double-bottom length (weeks, x5); 0 = off
        displayBoxes: true,        // i_displayBoxes
        displayHistoricalBoxes: false, // i_displayHistoricalBoxes
        pivotTie: 'both',          // see ASSUMPTION 1
        handles: true,             // NOT in Pine: look for a handle on recorded cups (false = Pine output)
        handleMinCupBars: 35,      // NOT in Pine: cup length old high -> now (7 weeks x5)
        handleMinCupDepthPct: 12,  // NOT in Pine: cup depth range for a handle (%)
        handleMaxCupDepthPct: 45,  // notes say "upwards of 33%", so not a hard 33; base detection itself allows 50
        handleMinUptrendPct: 30,   // NOT in Pine: real prior uptrend, old high vs lowest low of handleUptrendBars
        handleUptrendBars: 126,    // ~6 months up to and including the old high (uses fewer bars if the data is shorter)
        handleLipWithinPct: 15,    // right lip must be within this % of the old high
        handleMinBars: 5,          // minimum handle length (bars, 1 week)
        handleMinPullbackPct: 3,   // handle's lowest low must be at least this % below the lip high
        dbLegGapPct: 0.3,          // NOT in Pine: gap (% of price) between the drawn double-bottom W and the candles it clears
        dbLegKeepPct: 50,          // NOT in Pine: share of each double-bottom leg's original height that must be kept (0 = legs may flatten, 100 = full height)
        dbLegClear: 'wick',        // NOT in Pine: 'wick' = the W stays below every candle low; 'body' = only below the bodies (closer, crosses wick tips)
        priceScale: 'linear',      // NOT in Pine: 'log' when the chart is log-scaled, so the W is placed for the scale it is drawn on
        debug: false               // include pivots + per-bar flags in result
    };

    function assign(a, b) {
        for (var k in b) { if (Object.prototype.hasOwnProperty.call(b, k)) a[k] = b[k]; }
        return a;
    }

    function num(x) { return (typeof x === 'number' && isFinite(x)) ? x : NaN; }

    // Pine array.get(arr, k) on an array filled with array.unshift():
    // k = 0 is the most recently added value.
    function at(arr, k) {
        var v = arr[arr.length - 1 - k];
        return v === undefined ? NaN : v;
    }

    // Pine "for i = from to to by 1": inclusive, direction chosen from the
    // bounds (the sign of "by" is ignored). NaN bounds -> no iterations.
    function pineFor(from, to, fn) {
        var i;
        if (!isFinite(from) || !isFinite(to)) return;
        if (from <= to) { for (i = from; i <= to; i += 1) fn(i); }
        else            { for (i = from; i >= to; i -= 1) fn(i); }
    }

    // Smallest-total-shift solver for the double-bottom W. Solves
    //     minimise sum(s)   subject to   A s >= r,  s >= 0
    // (A has n columns, rows may have negative entries) through its dual
    //     maximise r.y      subject to   A' y <= 1,  y >= 0
    // whose all-slack starting point is feasible, so a plain one-phase simplex
    // (Bland's rule, so it cannot cycle) is enough. The primal answer is read
    // off the objective row under the slack columns. Returns null on failure.
    function minSumShifts(A, r, n) {
        var m = A.length, cols = m + n, i, j, f, piv, enter, leave, best, ratio, iter, T = [], row, obj = [], basis = [], s = [];
        var scale = 0;
        for (j = 0; j < m; j++) scale = Math.max(scale, Math.abs(r[j]));
        if (!(scale > 0)) scale = 1;
        for (i = 0; i < n; i++) {                       // one dual constraint per primal variable
            row = new Array(cols + 1);
            for (j = 0; j < m; j++) row[j] = A[j][i];
            for (j = 0; j < n; j++) row[m + j] = (i === j) ? 1 : 0;
            row[cols] = 1;
            T.push(row);
            basis.push(m + i);
        }
        for (j = 0; j < m; j++) obj.push(-r[j] / scale);
        for (j = m; j <= cols; j++) obj.push(0);
        for (iter = 0; iter < 5000; iter++) {
            enter = -1;
            for (j = 0; j < cols; j++) { if (obj[j] < -1e-12) { enter = j; break; } }
            if (enter < 0) break;
            leave = -1; best = Infinity;
            for (i = 0; i < n; i++) {
                if (T[i][enter] > 1e-12) {
                    ratio = T[i][cols] / T[i][enter];
                    if (ratio < best - 1e-15 || (Math.abs(ratio - best) <= 1e-15 && basis[i] < basis[leave])) { best = ratio; leave = i; }
                }
            }
            if (leave < 0) return null;
            piv = T[leave][enter];
            for (j = 0; j <= cols; j++) T[leave][j] /= piv;
            for (i = 0; i < n; i++) {
                if (i !== leave && T[i][enter] !== 0) {
                    f = T[i][enter];
                    for (j = 0; j <= cols; j++) T[i][j] -= f * T[leave][j];
                }
            }
            f = obj[enter];
            for (j = 0; j <= cols; j++) obj[j] -= f * T[leave][j];
            basis[leave] = enter;
        }
        if (iter >= 5000) return null;
        for (i = 0; i < n; i++) s.push(Math.max(0, obj[m + i]) * scale);
        return s;
    }

    function detect(ohlcv, userOpts) {
        var o = assign(assign({}, DEFAULTS), userOpts || {});
        var N = (ohlcv && ohlcv.length) ? ohlcv.length : 0;
        var res = { supported: true, bars: N, lines: [], boxes: [], cups: [], events: [], state: null };

        if (o.timeframe !== 'D') { res.supported = false; return res; }  // timeframe.isdaily

        var P = o.pivot | 0;
        var baseDepth  = o.baseDepthPct / 100;
        var baseLength = o.baseLengthWeeks * 5;
        var dbDepth    = o.dbDepthPct / 100;
        var dbLength   = o.dbLengthWeeks * 5;
        var dbMinLength = o.dbMinLengthWeeks * 5;
        if (N < 2 * P + 2) return res;

        // ── series ───────────────────────────────────────────────────────
        var T = new Array(N), H = new Array(N), L = new Array(N), C = new Array(N), O = new Array(N);
        var q, bar;
        for (q = 0; q < N; q++) {
            bar = ohlcv[q];
            T[q] = bar.time; H[q] = num(bar.high); L[q] = num(bar.low); C[q] = num(bar.close); O[q] = num(bar.open);
        }
        function hAt(i) { return (i >= 0 && i < N) ? H[i] : NaN; }
        function lAt(i) { return (i >= 0 && i < N) ? L[i] : NaN; }
        function cAt(i) { return (i >= 0 && i < N) ? C[i] : NaN; }

        // ta.highest / ta.lowest over the last len bars including bar n
        function wMax(arr, n, len) {
            if (n - len + 1 < 0) return NaN;
            var m = -Infinity, v;
            for (var k = n - len + 1; k <= n; k++) { v = arr[k]; if (v !== v) return NaN; if (v > m) m = v; }
            return m;
        }
        function wMin(arr, n, len) {
            if (n - len + 1 < 0) return NaN;
            var m = Infinity, v;
            for (var k = n - len + 1; k <= n; k++) { v = arr[k]; if (v !== v) return NaN; if (v < m) m = v; }
            return m;
        }

        // ta.pivothigh / ta.pivotlow candidate check (see ASSUMPTION 1)
        function pivotOK(isHigh, c) {
            var v = isHigh ? H[c] : L[c], w;
            if (v !== v) return false;
            for (var j = c - P; j <= c + P; j++) {
                if (j === c) continue;
                w = isHigh ? H[j] : L[j];
                if (w !== w) return false;
                if (isHigh ? (w > v) : (w < v)) return false;
                if (w === v) {
                    if (o.pivotTie === 'earlier' && j < c) return false;
                    if (o.pivotTie === 'later'   && j > c) return false;
                }
            }
            return true;
        }

        // ── persistent ("var") state ─────────────────────────────────────
        var pHP = [], pLP = [], pHB = [], pLB = [];  // pivots
        var startBasePrice = [], lowBasePrice = [];
        var lowOfBaseHigh = 0, startBaseBar = 0, lowerBaseBar = 0;
        var bottomDBPrice = [], topDBBar = 0, bottomDBBar = 0;
        var baseCond = false, baseCount = 0, lastLowestTrue = -1, prevDetectCup = false;
        var highLine = null, lowLine = null, dbLine1 = null, dbLine2 = null, dbLine3 = null;
        var entryBox = null, slBox = null, tpBox = null;

        // final per-bar values of Pine series, for the [1] history reads
        var sBase = new Array(N), sDB = new Array(N), sBox = new Array(N), sBoxOver = new Array(N);
        var sThird = new Array(N), sThirdTwo = new Array(N), sFourth = new Array(N), sFourthTwo = new Array(N);
        var dbgDetectCup = o.debug ? new Array(N) : null;
        var dbgBaseCond = o.debug ? new Array(N) : null;

        // ── drawing registry (what Pine would have on the chart) ─────────
        var lineList = [], boxList = [], cupList = [];
        function newLine(kind, x1, y1, x2, y2, dotted, width) {
            var ln = { kind: kind, x1: x1, y1: y1, x2: x2, y2: y2, dotted: dotted, width: width, deleted: false };
            lineList.push(ln);
            return ln;
        }
        function delLine(ln) { if (ln) ln.deleted = true; }
        function gy1(ln) { return ln ? ln.y1 : NaN; }
        function newBox(kind, left, top, right, bottom) {
            var bx = { kind: kind, left: left, top: top, right: right, bottom: bottom, deleted: false };
            boxList.push(bx);
            return bx;
        }
        function delBox(bx) { if (bx) bx.deleted = true; }
        function deleteBoxes() { delBox(entryBox); delBox(slBox); delBox(tpBox); }
        function ev(type, n, extra) {
            var e = { type: type, idx: n, time: T[n] };
            if (extra) assign(e, extra);
            res.events.push(e);
        }
        function redrawBaseLines(n) {
            var top = at(startBasePrice, 0), bot = at(lowBasePrice, 0);
            highLine = newLine('baseHigh', startBaseBar, top, n, top, true, 3);
            lowLine  = newLine('baseLow',  startBaseBar, bot, n, bot, false, 2);
        }
        function cupRecord(live, n) {
            var startUp = lowOfBaseHigh * 0.99;
            var bottom  = at(lowBasePrice, 0) * 0.99;
            var endUp   = lAt(n - 1) * 0.99;
            var left = lowerBaseBar - startBaseBar, right = n - lowerBaseBar;
            var valid = lowBasePrice.length > 0 && startUp > bottom && left >= 1 && right >= 1 && endUp === endUp;
            var rec = {
                live: live, valid: valid,
                startIdx: startBaseBar, lowIdx: lowerBaseBar, endIdx: n,
                tStart: T[startBaseBar], tLow: T[lowerBaseBar], tEnd: T[n],
                startUpPrice: startUp, bottomPrice: bottom, endUpPrice: endUp,
                lengthLeft: left, lengthRight: right,
                handle: null, handleReject: null
            };
            if (o.handles && valid) rec.handle = findHandle(rec, live, n);
            return rec;
        }

        // NOT in Pine: handle after the cup's right side (see CUP HANDLES under
        // DEVIATIONS FROM PINE). Returns the handle, or null when there is none.
        function findHandle(rec, live, n) {
            var sIdx = rec.startIdx, k;
            var oldHigh = H[sIdx];
            var cupLow = at(lowBasePrice, 0);
            // every rejection stores its reason on the cup (rec.handleReject)
            function no(msg) { rec.handleReject = msg; return null; }
            function pc(x) { return (x * 100).toFixed(1) + '%'; }
            if (!(oldHigh > 0) || !(cupLow === cupLow)) return no('no usable old high / cup low');

            // cup: length, depth, real prior uptrend
            if (n - sIdx < o.handleMinCupBars) return no('cup ' + (n - sIdx) + ' bars < ' + o.handleMinCupBars);
            var depth = 1 - cupLow / oldHigh;
            if (depth < o.handleMinCupDepthPct / 100 || depth > o.handleMaxCupDepthPct / 100) {
                return no('cup depth ' + pc(depth) + ' outside ' + o.handleMinCupDepthPct + '-' + o.handleMaxCupDepthPct + '%');
            }
            // lowest low of the bars up to and including the old high (clamped to the data we have)
            var priorLow = Infinity;
            for (k = Math.max(0, sIdx - o.handleUptrendBars + 1); k <= sIdx; k++) {
                if (L[k] < priorLow) priorLow = L[k];
            }
            if (!(priorLow < Infinity)) return no('no usable lows before the old high');
            var upX = oldHigh / priorLow, needX = 1 + o.handleMinUptrendPct / 100;
            if (upX < needX) return no('prior uptrend x' + upX.toFixed(2) + ' < x' + needX.toFixed(2));

            // Walk the right side bar by bar. A handle candidate is a running
            // peak (the right lip) plus the bars after it that stay at or below
            // it. A bar whose high takes out the peak ends that candidate on
            // the bar before (the pivot) and starts a new peak. Candidates that
            // pass the tests are kept; the latest one wins. (strict >, so the
            // first of equal highs is the lip.) For the reject reason, the
            // latest candidate that was long enough to be judged is reported.
            var middle = cupLow + (oldHigh - cupLow) * 0.5;
            var best = null, reason = null, lipIdx = -1, lipHigh = -Infinity;
            function tryHandle(endIdx) {
                if (endIdx - lipIdx < o.handleMinBars) return;
                if (!(lipHigh > middle)) { reason = 'lip below the middle of the cup'; return; }
                if (lipHigh < oldHigh * (1 - o.handleLipWithinPct / 100)) {
                    reason = 'lip ' + pc(1 - lipHigh / oldHigh) + ' below old high (max ' + o.handleLipWithinPct + '%)';
                    return;
                }
                var hLowIdx = -1, hLow = Infinity, j;
                for (j = lipIdx + 1; j <= endIdx; j++) {
                    if (L[j] < hLow) { hLow = L[j]; hLowIdx = j; }
                }
                if (hLowIdx < 0) { reason = 'no usable lows in the handle'; return; }
                var pullback = 1 - hLow / lipHigh;
                if (!(pullback >= o.handleMinPullbackPct / 100)) {
                    reason = 'handle pullback ' + pc(pullback) + ' < ' + o.handleMinPullbackPct + '%';
                    return;
                }
                if (!(hLow > middle)) { reason = 'handle low below the middle of the cup'; return; }
                best = {
                    lipIdx: lipIdx, tLip: T[lipIdx], lipHigh: lipHigh, lipPrice: L[lipIdx] * 0.99,
                    lowIdx: hLowIdx, tLow: T[hLowIdx], lowPrice: hLow * 0.99,
                    endIdx: endIdx, tEnd: T[endIdx], endPrice: L[endIdx] * 0.99,
                    length: endIdx - lipIdx
                };
            }
            for (k = rec.lowIdx + 1; k <= n; k++) {
                if (H[k] > lipHigh) {
                    if (lipIdx >= 0) tryHandle(k - 1);
                    lipIdx = k; lipHigh = H[k];
                }
            }
            if (lipIdx >= 0) tryHandle(n);   // candidate still open on the latest bar
            if (best) return best;
            return no(reason || 'no pullback of ' + o.handleMinBars + '+ bars after a right-side high');
        }

        // ── the bar loop: one pass == one Pine execution per bar ─────────
        for (var n = 0; n < N; n++) {
            var curH = H[n], curL = L[n];
            var isBase1 = n > 0 ? sBase[n - 1] : false;
            var isDB1 = n > 0 ? sDB[n - 1] : false;
            var isBox1 = n > 0 ? sBox[n - 1] : false;
            var boxOver1 = n > 0 ? sBoxOver[n - 1] : false;

            // PIVOTS HIGH/LOW
            if (n >= 2 * P) {
                var pc = n - P;
                if (pivotOK(true, pc)) {
                    pHP.push(H[pc]); pHB.push(pc);
                }
                if (pivotOK(false, pc)) {
                    pLP.push(L[pc]); pLB.push(pc);
                }
            }

            // BASE DETECTION CALCULATIONS
            var highestOfBase = wMax(H, n, 25);
            var highestOfComp = wMax(H, n, 50);
            var lowestBaseLow = wMin(L, n, 25);
            var lowestPointLeg = wMin(L, n, 103);
            var h25 = hAt(n - 25);

            if (pHP.length > 2) {
                var boolHighbase = h25 === at(pHP, 0) || h25 === at(pHP, 1) || h25 === at(pHP, 2);
                var boolHigherPiv = h25 >= highestOfComp;
                var legUpCond = h25 >= lowestPointLeg * 1.20;
                var firstBaseDepth = h25 * (1 - baseDepth) <= lowestBaseLow;
                var noCandleAbove = highestOfBase <= h25;
                var noBaseInBase = (n - 25 >= 0) ? sBase[n - 25] === false : false;
                baseCond = boolHighbase && boolHigherPiv && legUpCond && firstBaseDepth && noCandleAbove && noBaseInBase;
            }

            // ta.barssince(low == lowestBaseLow)
            if (curL === lowestBaseLow) lastLowestTrue = n;
            var lowestBar = lastLowestTrue < 0 ? NaN : n - lastLowestTrue;

            // If base detected, store key points
            var baseStart = baseCond && !isBase1;
            if (baseStart) {
                startBasePrice.push(h25);
                startBaseBar = n - 25;
                lowOfBaseHigh = lAt(n - 25);
                lowBasePrice.push(lowestBaseLow);
                lowerBaseBar = (lowestBar === lowestBar) ? n - lowestBar : NaN;
                ev('baseStart', n, { startIdx: startBaseBar, startTime: T[startBaseBar], top: h25, low: lowestBaseLow });
            }

            // isBase stays true while the base is still valid
            var isBase = false;
            if (baseCond || isBase1) isBase = true;

            // candle base counter
            if (baseStart) baseCount = n - startBaseBar;
            if (isBase) baseCount = baseCount + 1;

            // trade box state carried from the previous bar
            var isBox = false, boxOver = true;
            if (isBox1 && !boxOver1) isBox = true;

            // DOUBLE BOTTOM DETECTION
            var detectDB = false;
            if (isBase) {
                if (pHP.length > 1 && pLP.length > 1) {
                    var firstPivHigh = at(pHP, 1), scndPivHigh = at(pHP, 0);
                    var firstPivLow = at(pLP, 1), scndPivLow = at(pLP, 0);
                    var cA = firstPivHigh > scndPivHigh && firstPivLow * 0.97 > scndPivLow && firstPivLow < scndPivHigh;
                    var cB = scndPivLow >= (1 - dbDepth) * firstPivHigh;
                    var cC = scndPivLow <= 0.9 * firstPivHigh;
                    var cD = (firstPivHigh - scndPivLow) * 0.6 + scndPivLow <= scndPivHigh;
                    var cE = (firstPivHigh - scndPivLow) * 0.95 + scndPivLow >= scndPivHigh;
                    var denomF = scndPivHigh - scndPivLow;
                    var cF = denomF !== 0 && (firstPivHigh - firstPivLow) / denomF >= 0.70;   // ASSUMPTION 4
                    var cG = (firstPivHigh - firstPivLow) / 2 + firstPivLow <= scndPivHigh;
                    var cH = true;
                    if (startBasePrice.length > 0) cH = firstPivHigh === at(startBasePrice, 0);

                    var t1 = at(pHB, 1), t2 = at(pLB, 1), t3 = at(pHB, 0), t4 = at(pLB, 0);
                    var tA = t1 < t2 && t2 < t3 && t3 < t4;
                    var tB = t4 - t1 <= 100;
                    var tC = (t3 - t1) * 2 >= n - t3;
                    var tD = t3 - t1 <= (n - t3) * 2;
                    var tE = t2 - t1 <= 2 * (t3 - t2) && t3 - t2 <= (t2 - t1) * 2;
                    var tF = t4 - t3 <= 2 * (t3 - t2) && t4 - t3 >= 0.5 * (t3 - t2);
                    var tG = n - t1 >= dbMinLength;   // NOT in Pine: minimum length (old high -> now), 7 weeks by default

                    if (cA && cB && cC && cD && cE && cF && cG && cH && tA && tB && tC && tD && tE && tF && tG) {
                        // condBothA = ta.highest(high, bar_index - frthPivTime) <= scndPivHigh
                        var spanMax = wMax(H, n, n - t4);
                        if (spanMax <= scndPivHigh) {
                            detectDB = true;
                            bottomDBPrice.push(scndPivLow);
                            topDBBar = at(pHB, 1);
                        }
                    }
                }
            }

            var isDoubleBottom = false;
            if (detectDB || isDB1) isDoubleBottom = true;

            // CUP DETECTION
            var detectCup = false;
            var condThird = false, condFourth = false, condThirdTwo = false, condFourthTwo = false;
            if (isBase) {
                var highCup = 0, lowCup = 0;
                if (startBasePrice.length > 0 && lowBasePrice.length > 0) {
                    highCup = at(startBasePrice, 0);
                    lowCup = at(lowBasePrice, 0);
                }
                var middleOfCup = lowCup + (highCup - lowCup) * 0.5;
                var condMaxDepth = lowCup >= (1 - baseDepth) * highCup;
                var condMinDepth = lowCup <= 0.92 * highCup;
                var condMaxLength = n - startBaseBar <= baseLength;
                var condMinLength = n - startBaseBar >= 30;
                var condAbsolutePos = (highCup - lowCup) * 0.5 + lowCup <= curH;

                var baseTier = baseCount / 3, baseFourth = baseCount / 4;
                var cnt = 0;
                // middle third: closes <= middle of the cup
                pineFor(baseCount - baseTier, baseTier, function(i) { if (cAt(n - Math.floor(i)) <= middleOfCup) cnt++; });
                condThird = cnt / baseTier >= 0.90;
                // oldest quarter: closes >= middle of the cup (the effective condThirdTwo)
                cnt = 0;
                pineFor(baseCount, 3 * baseFourth, function(i) { if (cAt(n - Math.floor(i)) >= middleOfCup) cnt++; });
                condThirdTwo = cnt / baseFourth >= 0.3;
                // middle half: closes <= middle of the cup
                cnt = 0;
                pineFor(baseCount - baseFourth, baseFourth, function(i) { if (cAt(n - Math.floor(i)) <= middleOfCup) cnt++; });
                condFourth = cnt / (2 * baseFourth) >= 0.90;
                // condFourthTwo is never assigned in the Pine script -> stays false

                var p1 = n > 0;
                var cupForm = (condThird && p1 && sThird[n - 1] && condThirdTwo && sThirdTwo[n - 1]) ||
                              (condFourth && p1 && sFourth[n - 1] && condFourthTwo && sFourthTwo[n - 1]);
                detectCup = !!(condMaxDepth && condMinDepth && condMaxLength && condMinLength && condAbsolutePos && cupForm);
            }
            if (o.debug) { dbgDetectCup[n] = detectCup; dbgBaseCond[n] = baseCond; }
            if (detectCup && !prevDetectCup) ev('cupDetected', n, { baseCount: baseCount });
            prevDetectCup = detectCup;

            // DRAWING BASE ON CHART IF DETECTION
            if (baseStart) {
                redrawBaseLines(n);
                var select = o.displayHistoricalBoxes ? false : boxOver;
                if (select) { deleteBoxes(); isBox = false; boxOver = true; }
            }

            // Exception with ta.lowest() series not equal to lower low (eg IOT)
            if (lowerBaseBar - startBaseBar <= 0) {
                var lowValue = curL;
                pineFor(baseCount, 0, function(i) {
                    var lv = lAt(n - i);
                    if (lv < lowValue) { lowerBaseBar = n - i; lowValue = lv; }
                });
            }

            // DRAWING CURRENT CUP (barstate.islast only)
            if (n === N - 1 && detectCup && isBase && !isDB1) {
                delLine(lowLine);   // Pine: line.delete(lowLine) before drawing the live cup
                cupList.push(cupRecord(true, n));
            }

            // DRAWING DOUBLE BOTTOM ON CHART IF DETECTION
            if (detectDB && !isDB1) {
                delLine(highLine); delLine(lowLine);
                bottomDBBar = at(pLB, 0);
                var dbTop = at(pHP, 0);
                highLine = newLine('dbHigh', at(pHB, 0), dbTop, n, dbTop, true, 3);
                // y of each peak = low of that pivot-high candle x 0.99 (see DEVIATIONS FROM PINE)
                dbLine1 = newLine('dbLeg1', at(pHB, 1), lAt(at(pHB, 1)) * 0.99, at(pLB, 1), at(pLP, 1) * 0.99, false, 2);
                dbLine2 = newLine('dbLeg2', at(pLB, 1), at(pLP, 1) * 0.99, at(pHB, 0), lAt(at(pHB, 0)) * 0.99, false, 2);
                dbLine3 = newLine('dbLeg3', at(pHB, 0), lAt(at(pHB, 0)) * 0.99, bottomDBBar, at(bottomDBPrice, 0) * 0.99, false, 2);
                lowLine = newLine('dbLow', bottomDBBar, at(bottomDBPrice, 0) * 0.99, n, curL * 0.99, false, 2);
                var select2 = o.displayHistoricalBoxes ? false : boxOver;
                if (select2) { deleteBoxes(); isBox = false; boxOver = true; }
                ev('doubleBottom', n, { firstHighIdx: at(pHB, 1), lowIdx: at(pLB, 1), secondHighIdx: at(pHB, 0), secondLowIdx: bottomDBBar });
            }

            // Check Bottom Break: adjust double bottom base
            if (bottomDBPrice.length > 0) {
                if (curL < at(bottomDBPrice, 0) && isDoubleBottom) {
                    delLine(highLine); delLine(lowLine); delLine(dbLine1); delLine(dbLine2); delLine(dbLine3);
                    isDoubleBottom = false;
                    redrawBaseLines(n);   // redraw original base if double bottom disabled
                    ev('doubleBottomEnd', n, { reason: 'bottomBreak' });
                }
            }
            // Check Double Bottom base length
            if (isDoubleBottom && n - topDBBar > dbLength) {
                delLine(highLine); delLine(lowLine); delLine(dbLine1); delLine(dbLine2); delLine(dbLine3);
                isDoubleBottom = false;
                ev('doubleBottomEnd', n, { reason: 'length' });
            }
            // Check Double Bottom pattern properties  ( (isDB and A) or B  -- Pine precedence )
            var firstPivTime = dbLine1 ? dbLine1.x1 : NaN;
            var thrdPivTime = dbLine2 ? dbLine2.x2 : NaN;
            if ((isDoubleBottom && (thrdPivTime - firstPivTime) * 2 <= n - thrdPivTime) ||
                (thrdPivTime - firstPivTime >= (n - thrdPivTime) * 2)) {
                delLine(highLine); delLine(lowLine); delLine(dbLine1); delLine(dbLine2); delLine(dbLine3);
                var wasDB = isDoubleBottom;
                isDoubleBottom = false;
                redrawBaseLines(n);       // redraw original base if double bottom disabled
                if (wasDB) ev('doubleBottomEnd', n, { reason: 'proportions' });
            }

            // Adjust Base Bottom
            if (curL < gy1(lowLine) && isBase && !isDoubleBottom) {
                lowLine.y1 = curL; lowLine.y2 = curL;
                lowerBaseBar = n;
                if (lowBasePrice.length > 0) lowBasePrice.push(curL);
            }
            // Check Base Max Depth and Max Length
            if ((curL < gy1(highLine) * (1 - baseDepth) || baseCount > baseLength) && isBase1 && !isDB1) {
                delLine(highLine); delLine(lowLine);
                isBase = false;
                ev('baseEnd', n, { reason: baseCount > baseLength ? 'length' : 'depth' });
            }
            // Check depth of base to adjust style of line
            if (gy1(highLine) * 0.85 >= gy1(lowLine) && isBase && !isDoubleBottom) {
                lowLine.dotted = true; lowLine.width = 3;
            }
            // While the base is still valid, we extend lines
            if (curH <= gy1(highLine) && curL >= gy1(lowLine) && isBase) {
                highLine.x2 = n; lowLine.x2 = n;
                if (isDoubleBottom) { lowLine.y2 = curL * 0.99; lowLine.x2 = n; }
            }

            // Check Top Break
            if (curH > gy1(highLine) && (isBase1 || isDB1)) {
                isBase = false;
                if (isDoubleBottom) isDoubleBottom = false;
                var cupAtBreak = false;
                // Drawing historical cups if they are cups during the break out
                if (detectCup && !isDB1) {
                    delLine(lowLine);
                    if (lowerBaseBar - startBaseBar <= 0) {
                        var lowValue2 = curL;
                        pineFor(baseCount - 1, 0, function(i) {
                            var lv2 = lAt(n - i);
                            if (lv2 < lowValue2) { lowerBaseBar = n - i; lowValue2 = lv2; }
                        });
                    }
                    var rec = cupRecord(false, n);
                    if (rec.valid) { cupList.push(rec); cupAtBreak = true; }
                }
                if (o.displayBoxes) {
                    var topBase = gy1(highLine);
                    entryBox = newBox('entry',      n - 1, topBase * 1.05, n, topBase);
                    slBox    = newBox('stopLoss',   n - 1, topBase * 0.95, n, topBase * 0.92);
                    tpBox    = newBox('takeProfit', n - 1, topBase * 1.20, n, topBase * 1.25);
                    isBox = true;
                    boxOver = false;
                }
                ev('breakout', n, { top: gy1(highLine), fromDoubleBottom: isDB1, cup: cupAtBreak });
            }

            // TRADE BOX GESTION: extend boxes while the trade runs
            if (curH <= gy1(highLine) * 1.25 && curL >= gy1(highLine) * 0.92 && isBox) {
                if (entryBox) entryBox.right = n;
                if (slBox) slBox.right = n;
                if (tpBox) tpBox.right = n;
                isBox = true;
                boxOver = false;
            }

            // store final values of this bar for the [1] reads of the next bar
            sBase[n] = isBase; sDB[n] = isDoubleBottom; sBox[n] = isBox; sBoxOver[n] = boxOver;
            sThird[n] = condThird; sThirdTwo[n] = condThirdTwo; sFourth[n] = condFourth; sFourthTwo[n] = condFourthTwo;
        }

        // ── handle cups: the buy point is the handle high ────────────────
        // NOT in Pine (see BUY POINT under CUP HANDLES in the header). The handle
        // is only found when the cup is recorded, which is after the bars it
        // describes, so this is done here on what the bar loop produced instead
        // of inside it. For every cup with a handle:
        //  - the old-high dotted line ('baseHigh') is dropped and a 'handleHigh'
        //    dotted line is drawn on the lip high, from the lip bar to the bar
        //    that takes it out (or to the latest bar while the handle is open)
        //  - the old-high 'breakout' event is removed and a 'breakout' event
        //    (handle: true) is put on the bar that takes the lip high out
        //  - the entry / stop / target boxes start on that bar and are anchored
        //    to the lip high, extended with the same rule Pine uses (while the
        //    bar stays inside 0.92x .. 1.25x of the buy point). They are only
        //    drawn where the old-high boxes would still have been on the chart
        //    (or for a live cup that has not reached the old high yet)
        (function applyHandleBuyPoints() {
            var done = {}, ci, k, cp, hd, key, top, brk, broke, hadOldBreak, survived, right, kept, ev1;
            for (ci = 0; ci < cupList.length; ci++) {
                cp = cupList[ci]; hd = cp.handle;
                if (!cp.valid || !hd) continue;
                key = cp.startIdx + '|' + hd.lipIdx;
                if (done[key]) continue;                  // a cup can be recorded twice on the breakout bar
                done[key] = true;
                top = hd.lipHigh;
                brk = hd.endIdx + 1;                      // first bar that takes the lip high out
                broke = brk <= cp.endIdx;                 // false while the handle is still open on the latest bar

                // dotted line
                for (k = 0; k < lineList.length; k++) {
                    if (lineList[k].kind === 'baseHigh' && lineList[k].x1 === cp.startIdx) lineList[k].deleted = true;
                }
                newLine('handleHigh', hd.lipIdx, top, broke ? brk : cp.endIdx, top, true, 3);

                // the old-high breakout of this cup (event + boxes), if price got that far
                hadOldBreak = false; kept = [];
                for (k = 0; k < res.events.length; k++) {
                    ev1 = res.events[k];
                    if (ev1.type === 'breakout' && ev1.cup && ev1.idx === cp.endIdx) hadOldBreak = true;
                    else kept.push(ev1);
                }
                res.events = kept;
                survived = !hadOldBreak;                  // live cup still inside the base: nothing was ever drawn
                if (hadOldBreak) {
                    for (k = 0; k < boxList.length; k++) {
                        if (boxList[k].left === cp.endIdx - 1) {
                            if (!boxList[k].deleted) survived = true;
                            boxList[k].deleted = true;
                        }
                    }
                }
                if (!broke) continue;

                res.events.push({ type: 'breakout', idx: brk, time: T[brk], top: top, fromDoubleBottom: false, cup: true, handle: true });
                if (o.displayBoxes && survived) {
                    right = brk;
                    for (k = brk; k < N; k++) {
                        if (H[k] <= top * 1.25 && L[k] >= top * 0.92) right = k; else break;
                    }
                    newBox('entry',      brk - 1, top * 1.05, right, top);
                    newBox('stopLoss',   brk - 1, top * 0.95, right, top * 0.92);
                    newBox('takeProfit', brk - 1, top * 1.20, right, top * 1.25);
                }
            }
            // the re-anchored breakout events can now be out of order: sort by bar (stable)
            res.events = res.events.map(function(e, i) { return { e: e, i: i }; })
                .sort(function(x, y) { return (x.e.idx - y.e.idx) || (x.i - y.i); })
                .map(function(x) { return x.e; });
        })();

        // ── build the result ─────────────────────────────────────────────
        function tm(i) { return (i === i && i >= 0 && i < N) ? T[Math.round(i)] : null; }
        var i, ln, bx;
        for (i = 0; i < lineList.length; i++) {
            ln = lineList[i];
            if (ln.deleted) continue;
            if (!(ln.y1 === ln.y1 && ln.y2 === ln.y2 && tm(ln.x1) !== null && tm(ln.x2) !== null)) continue;
            res.lines.push({ kind: ln.kind, t1: tm(ln.x1), p1: ln.y1, t2: tm(ln.x2), p2: ln.y2,
                             idx1: ln.x1, idx2: ln.x2, dotted: ln.dotted, width: ln.width });
        }
        for (i = 0; i < boxList.length; i++) {
            bx = boxList[i];
            if (bx.deleted) continue;
            res.boxes.push({ kind: bx.kind, t1: tm(bx.left), t2: tm(bx.right), idx1: bx.left, idx2: bx.right,
                             top: Math.max(bx.top, bx.bottom), bottom: Math.min(bx.top, bx.bottom) });
        }
        res.cups = cupList;

        // ── double bottoms: keep the W below the candles ─────────────────
        // NOT in Pine (see DOUBLE BOTTOM LEGS in the header). The four legs are
        // created with their vertices on the low of each vertex candle (x 0.99),
        // so a straight leg can run through the candles in between. Here the five
        // vertices are placed again, each one as high as the others allow, so
        // that every leg is below every candle it spans:
        //   - each vertex starts on the clear level of its own candle (its low, or
        //     body, minus opts.dbLegGapPct) and may only move DOWN from there
        //   - every candle under a leg forces the two vertices of that leg down far
        //     enough to clear it
        //   - every leg has to keep at least opts.dbLegKeepPct of its original
        //     height, so a steep drop after the middle peak cannot flatten the W
        //   - among all placements that satisfy this, the one with the smallest
        //     total downward movement is used (an exact linear programme)
        // Done in the scale the chart is drawn on (opts.priceScale), because a
        // straight line on screen is straight in that scale, not in price. If the
        // solver ever fails or returns something that does not verify, all five
        // vertices are moved down by the same amount instead, which always clears.
        // This only edits the OUTPUT lines (res.lines), after they were built, so
        // nothing the detector reports (state, events, cups) can change.
        (function placeDoubleBottomLegs() {
            var log = o.priceScale === 'log';
            var tr  = log ? Math.log : function(v) { return v; };
            var inv = log ? Math.exp : function(v) { return v; };
            var keep = 1 - o.dbLegGapPct / 100;
            var alpha = Math.min(1, Math.max(0, o.dbLegKeepPct / 100));
            function find(kind, idx1) {
                for (var q2 = 0; q2 < res.lines.length; q2++) {
                    if (res.lines[q2].kind === kind && res.lines[q2].idx1 === idx1) return res.lines[q2];
                }
                return null;
            }
            function clearLevel(k) {          // the price the W has to stay below on bar k
                var v = L[k], bd;
                if (o.dbLegClear === 'body') { bd = Math.min(O[k], C[k]); if (bd === bd) v = bd; }
                return v;
            }
            function positive(v) { return v === v && v > 0; }
            function feasibleShifts(sv, rows, rhs, tol) {   // does sv satisfy sv >= 0 and every row?
                var q3, q4, tot;
                if (!sv) return false;
                for (q3 = 0; q3 < 5; q3++) { if (!(sv[q3] >= 0) || !isFinite(sv[q3])) return false; }
                for (q3 = 0; q3 < rows.length; q3++) {
                    tot = 0;
                    for (q4 = 0; q4 < 5; q4++) tot += rows[q3][q4] * sv[q4];
                    if (tot < rhs[q3] - tol) return false;
                }
                return true;
            }
            var floors = [[1, 0], [1, 2], [3, 2], [3, 4]];     // [trough, the vertex it hangs from]
            var li, a, b, c, d, xs, base, rows, rhs, seg, k, t, lv, rk, j, D, row, sh, uni, ok, tol, scale, ny, i2;
            for (li = 0; li < res.lines.length; li++) {
                a = res.lines[li];
                if (a.kind !== 'dbLeg1') continue;
                b = find('dbLeg2', a.idx2);
                c = b && find('dbLeg3', b.idx2);
                d = c && find('dbLow', c.idx2);
                if (!d) continue;
                xs = [a.idx1, a.idx2, b.idx2, c.idx2, d.idx2];   // the five vertices of the W (bars)
                if (!(xs[1] > xs[0] && xs[2] > xs[1] && xs[3] > xs[2] && xs[4] > xs[3])) continue;
                base = []; ok = true;
                for (j = 0; j < 5; j++) {
                    lv = clearLevel(xs[j]);
                    if (!positive(lv)) { ok = false; break; }
                    base.push(tr(lv * keep));
                }
                if (!ok) continue;
                rows = []; rhs = []; uni = 0; scale = 0;
                for (seg = 0; seg < 4; seg++) {
                    for (k = xs[seg]; k <= xs[seg + 1]; k++) {
                        lv = clearLevel(k);
                        if (!positive(lv)) continue;
                        t = (k - xs[seg]) / (xs[seg + 1] - xs[seg]);
                        rk = (1 - t) * base[seg] + t * base[seg + 1] - tr(lv * keep);   // how far the leg would sit above this candle
                        scale = Math.max(scale, Math.abs(rk));
                        if (rk > 0) {
                            row = [0, 0, 0, 0, 0]; row[seg] = 1 - t; row[seg + 1] = t;
                            rows.push(row); rhs.push(rk); uni = Math.max(uni, rk);
                        }
                    }
                }
                for (j = 0; j < floors.length; j++) {
                    D = Math.max(base[floors[j][1]] - base[floors[j][0]], 0);
                    row = [0, 0, 0, 0, 0]; row[floors[j][0]] = 1; row[floors[j][1]] = -1;
                    rows.push(row); rhs.push(-(1 - alpha) * D);
                    scale = Math.max(scale, D);
                }
                tol = 1e-9 * (scale > 0 ? scale : 1);
                sh = rows.length ? minSumShifts(rows, rhs, 5) : [0, 0, 0, 0, 0];
                if (!feasibleShifts(sh, rows, rhs, tol)) sh = [uni, uni, uni, uni, uni];
                ny = [];
                for (i2 = 0; i2 < 5; i2++) ny.push(inv(base[i2] - sh[i2]));
                if (!ny.every(positive)) continue;
                a.p1 = ny[0]; a.p2 = ny[1];
                b.p1 = ny[1]; b.p2 = ny[2];
                c.p1 = ny[2]; c.p2 = ny[3];
                d.p1 = ny[3]; d.p2 = ny[4];
            }
        })();

        res.state = {
            isBase: sBase[N - 1], isDoubleBottom: sDB[N - 1], inTradeBox: sBox[N - 1], baseCount: baseCount,
            baseStartIdx: sBase[N - 1] ? startBaseBar : null,
            baseTop: sBase[N - 1] ? gy1(highLine) : null,
            baseLow: sBase[N - 1] ? gy1(lowLine) : null
        };
        if (o.debug) {
            res.debug = { pivotHighIdx: pHB.slice(), pivotHighPrice: pHP.slice(), pivotLowIdx: pLB.slice(), pivotLowPrice: pLP.slice(),
                          isBase: sBase, isDoubleBottom: sDB, detectCup: dbgDetectCup, baseCond: dbgBaseCond };
        }
        return res;
    }

    // Polyline for a cup record, exactly the exp() curve the Pine script draws
    // (one point per bar instead of one line object per bar).
    // Returns [{idx, price}, ...] left-to-right.
    function cupCurve(cup) {
        var pts = [], i;
        if (!cup || !cup.valid) return pts;
        var Lf = cup.lengthLeft, Rt = cup.lengthRight;
        var sU = cup.startUpPrice, b = cup.bottomPrice, eU = cup.endUpPrice;
        // NOT in Pine: with a handle the right side stops at the right lip
        if (cup.handle) { Rt = cup.handle.lipIdx - cup.lowIdx; eU = cup.handle.lipPrice; }
        for (i = 0; i <= Lf; i++) {
            pts.push({ idx: cup.startIdx + i, price: i === Lf ? b : b + (sU - b) / Math.exp(i * 6 / Lf) });
        }
        for (i = 1; i <= Rt; i++) {
            pts.push({ idx: cup.lowIdx + i, price: b + (Math.exp(i * 6 / Rt) - 1) * (eU - b) / Math.exp(6) });
        }
        return pts;
    }

    // NOT in Pine: a cup's handle, drawn as one straight line sloping down from
    // the lip candle's low to the handle's lowest low (the decline only; the
    // move back up is not drawn), both at 0.99 x the low like the cup.
    // Returns [] when the cup has no handle.
    function handleCurve(cup) {
        var h = (cup && cup.valid) ? cup.handle : null;
        if (!h) return [];
        return [{ idx: h.lipIdx, price: h.lipPrice }, { idx: h.lowIdx, price: h.lowPrice }];
    }

    var api = { detect: detect, cupCurve: cupCurve, handleCurve: handleCurve, DEFAULTS: DEFAULTS, VERSION: '0.3.6-db-legs-lp' };
    if (typeof window !== 'undefined') window.PatternDetector = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
