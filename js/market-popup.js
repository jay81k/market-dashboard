// ── Macro card hover chart popup (Lightweight Charts + Yahoo Finance) ─────
var _mmPopup = (function () {
    var popup      = null;
    var popupSym   = null;
    var popupChart = null;
    var popupChg   = null;
    var popupCaret = null;
    var popupIndRank   = null;
    var popupRsBadge   = null;
    var popup3mrsBadge = null;
    var popupName    = null;
    var popupIndName = null;
    var popupIndSep  = null;

    var hoverTimer        = null;
    var hideTimer         = null;
    var liveRefreshTimer  = null;
    var lwChart     = null;
    var candleSeries = null;
    var activeCard  = null;
    var activeYfSym = null;   // YF symbol for macro cards (e.g. 'CL=F'); null for regular tickers
    var POPUP_W     = 420;
    var POPUP_H     = 310;
    var ready       = false;

    function init() {
        popup        = document.getElementById('mm-hover-popup');
        popupSym     = document.getElementById('mm-popup-sym');
        popupChart   = document.getElementById('mm-popup-chart');
        popupChg     = document.getElementById('mm-popup-chg');
        popupCaret   = document.getElementById('mm-popup-caret');
        popupIndRank   = document.getElementById('mm-popup-ind-rank');
        popupRsBadge   = document.getElementById('mm-popup-rs-badge');
        popup3mrsBadge = document.getElementById('mm-popup-3mrs-badge');
        popupName      = document.getElementById('mm-popup-name');
        popupIndName   = document.getElementById('mm-popup-ind-name');
        popupIndSep    = document.getElementById('mm-popup-ind-sep');
        if (!popup) return;

        // mousemove tracker handles hide logic — no listeners needed on popup itself
        window.addEventListener('resize', hidePopup);
        // Close immediately when user scrolls anywhere
        window.addEventListener('wheel', function() {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            hidePopup();
        }, { passive: true });

        // Left-click anywhere in popup → close popup then open fullscreen chart modal
        popup.addEventListener('click', function(e) {
            var ticker = popupSym ? popupSym.textContent.trim() : '';
            if (!ticker || ticker === '—') return;
            hidePopup();
            if (typeof openMcFullscreen === 'function') {
                if (activeYfSym) {
                    openMcFullscreen(activeYfSym, undefined, ticker);
                } else {
                    openMcFullscreen(ticker);
                }
            }
        });

        // Right-click anywhere in popup → watchlist / alert picker
        popup.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            e.stopPropagation();
            var ticker = popupSym ? popupSym.textContent.trim() : '';
            if (!ticker || ticker === '—') return;
            var fakeBtn = {
                getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
                getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                _wlNoSwitch: true
            };
            wlOpenPicker(fakeBtn, e, false);
            wlPickerJustOpened = false;
        });

        ready = true;
    }

    function destroyChart() {
        if (lwChart) {
            try { lwChart.remove(); } catch(e) {}
            lwChart = null;
            candleSeries = null;
        }
        if (popupChart) popupChart.innerHTML = '';
    }

    function positionPopup(card) {
        var panelEl   = card.closest('#scan-nav-panel') || card.closest('.wl-side-panel');
        var tableWrap = card.closest('.stocks-table-wrap') || card.closest('#al-list');

        popup.style.width = POPUP_W + 'px';

        if (panelEl) {
            // ── Side placement: popup appears to the LEFT of the scan nav / watchlist panel ──
            var panelRect = panelEl.getBoundingClientRect();
            var rowRect   = card.getBoundingClientRect();

            var left = Math.max(8, panelRect.left - POPUP_W - 10);
            popup.style.left   = left + 'px';
            popup.style.right  = '';

            var centerY = rowRect.top + rowRect.height / 2;
            var top     = Math.max(8, Math.min(centerY - POPUP_H / 2, window.innerHeight - POPUP_H - 8));
            popup.style.top    = top + 'px';
            popup.style.bottom = '';

            // Right-pointing caret aligned to the hovered row
            popupCaret.className       = 'mm-popup-caret right';
            popupCaret.style.left      = '';
            popupCaret.style.bottom    = '';
            popupCaret.style.transform = '';
            var caretTop = (centerY - top) - 8;
            popupCaret.style.top = Math.max(16, Math.min(caretTop, POPUP_H - 16)) + 'px';
            return;
        }

        if (tableWrap) {
            // ── Side placement: popup appears to the RIGHT of the ticker cell ──
            var rowRect  = card.getBoundingClientRect();
            var left     = rowRect.right + 10;
            if (left + POPUP_W > window.innerWidth - 8) left = window.innerWidth - POPUP_W - 8;
            popup.style.left   = left + 'px';
            popup.style.right  = '';

            var centerY = rowRect.top + rowRect.height / 2;
            var top     = Math.max(8, Math.min(centerY - POPUP_H / 2, window.innerHeight - POPUP_H - 8));
            popup.style.top    = top + 'px';
            popup.style.bottom = '';

            // Left-pointing caret aligned to the hovered row
            popupCaret.className       = 'mm-popup-caret left';
            popupCaret.style.right     = '';
            popupCaret.style.bottom    = '';
            popupCaret.style.transform = '';
            var caretTop = (centerY - top) - 8;
            popupCaret.style.top = Math.max(16, Math.min(caretTop, POPUP_H - 16)) + 'px';
            return;
        }

        // ── Default: above / below placement ──
        var rect       = card.getBoundingClientRect();
        var spaceAbove = rect.top;
        var spaceBelow = window.innerHeight - rect.bottom;
        var placeAbove = spaceAbove > POPUP_H + 16 || spaceAbove >= spaceBelow;

        var left = rect.left + (rect.width / 2) - (POPUP_W / 2);
        left = Math.max(8, Math.min(left, window.innerWidth - POPUP_W - 8));
        popup.style.left = left + 'px';

        if (placeAbove) {
            popup.style.top    = '';
            popup.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
            popupCaret.className      = 'mm-popup-caret down';
            popupCaret.style.bottom   = '-8px';
            popupCaret.style.top      = '';
        } else {
            popup.style.bottom = '';
            popup.style.top    = (rect.bottom + 8) + 'px';
            popupCaret.className    = 'mm-popup-caret up';
            popupCaret.style.top    = '-8px';
            popupCaret.style.bottom = '';
        }
        var caretLeft = (rect.left + rect.width / 2) - left;
        popupCaret.style.left      = Math.max(16, Math.min(caretLeft, POPUP_W - 16)) + 'px';
        popupCaret.style.transform = 'none';
    }

    function renderChart(ohlcv) {
        destroyChart();
        if (!LightweightCharts) return;

        lwChart = LightweightCharts.createChart(popupChart, {
            width:  popupChart.clientWidth  || POPUP_W,
            height: popupChart.clientHeight || 220,
            layout: { background: { color: '#0d1117' }, textColor: '#6e7681' },
            grid:   { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
            rightPriceScale: { borderColor: '#21262d', textColor: '#6e7681' },
            timeScale: { borderColor: '#21262d', timeVisible: false },
            handleScroll: false,
            handleScale:  false,
        });

        candleSeries = lwChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor:          '#3fb950',
            downColor:        '#f85149',
            borderVisible:    false,
            wickUpColor:      '#3fb950',
            wickDownColor:    '#f85149',
            priceLineVisible: false,
            lastValueVisible: true,
        });

        candleSeries.setData(ohlcv);

        // SMA 50
        var sma50 = lwChart.addSeries(LightweightCharts.LineSeries, {
            color:           '#f23645',
            lineWidth:       1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });
        var sma50data = [];
        for (var i = 49; i < ohlcv.length; i++) {
            var sum = 0;
            for (var j = i - 49; j <= i; j++) sum += ohlcv[j].close;
            sma50data.push({ time: ohlcv[i].time, value: sum / 50 });
        }
        sma50.setData(sma50data);

        // EMA 21
        var ema21 = lwChart.addSeries(LightweightCharts.LineSeries, {
            color:           '#2979c8',
            lineWidth:       1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });
        var ema21data = [];
        var k = 2 / (21 + 1);
        var emaVal = ohlcv[0].close;
        for (var i = 0; i < ohlcv.length; i++) {
            emaVal = ohlcv[i].close * k + emaVal * (1 - k);
            if (i >= 20) ema21data.push({ time: ohlcv[i].time, value: emaVal });
        }
        ema21.setData(ema21data);

        // Volume
        var volSeries = lwChart.addSeries(LightweightCharts.HistogramSeries, {
            priceFormat:  { type: 'volume' },
            priceScaleId: 'volume',
        });
        lwChart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });
        volSeries.setData(ohlcv.map(function(d) {
            return {
                time:  d.time,
                value: d.volume,
                color: d.close >= d.open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)',
            };
        }));

        lwChart.timeScale().fitContent();
        var totalBars = ohlcv.length;
        lwChart.timeScale().setVisibleLogicalRange({ from: totalBars - 65, to: totalBars - 1 });

        // OHLC crosshair readout
        var ohlcLegend = document.createElement('div');
        ohlcLegend.style.cssText = [
            'position:absolute', 'top:-10px', 'left:6px', 'z-index:10',
            'font-size:11px', 'font-weight:600', 'font-variant-numeric:tabular-nums',
            'color:#8b949e', 'pointer-events:none', 'line-height:1.5',
            'background:rgba(13,17,23,0.7)', 'padding:2px 6px', 'border-radius:3px',
        ].join(';');
        popupChart.style.position = 'relative';
        popupChart.appendChild(ohlcLegend);

        function fmtP(v) { return v != null ? v.toFixed(2) : '—'; }
        function fmtV(v) {
            if (v == null) return '—';
            return v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : v.toFixed(0);
        }

        lwChart.subscribeCrosshairMove(function(param) {
            if (!param.time || !param.seriesData || !param.seriesData.size) {
                ohlcLegend.innerHTML = '';
                return;
            }
            var d = param.seriesData.get(candleSeries);
            if (!d) { ohlcLegend.innerHTML = ''; return; }
            var cl   = d.close >= d.open ? '#3fb950' : '#f85149';
            var vd   = param.seriesData.get(volSeries);
            var volStr = vd ? '&nbsp;&nbsp;<span style="color:#484f58">Vol</span> ' + fmtV(vd.value) : '';
            ohlcLegend.innerHTML =
                '<span style="color:#6e7681">O</span> <span style="color:' + cl + '">' + fmtP(d.open)  + '</span>&nbsp; ' +
                '<span style="color:#6e7681">H</span> <span style="color:' + cl + '">' + fmtP(d.high)  + '</span>&nbsp; ' +
                '<span style="color:#6e7681">L</span> <span style="color:' + cl + '">' + fmtP(d.low)   + '</span>&nbsp; ' +
                '<span style="color:#6e7681">C</span> <span style="color:' + cl + '">' + fmtP(d.close) + '</span>' +
                volStr;
        });
    }

    function appendTodayCandle(sym, activeRef, todayTs) {
        // Skip rather than pile onto an active shared cooldown — this is a
        // background refresh, not user-visible loading state, so silently
        // skipping this round is the safe default (it'll catch up next hover).
        if (window.yahooProxyPace && Date.now() < window.yahooProxyPace.cooldownUntil()) return;
        var url = 'https://yahoo-proxy.jay69k.workers.dev?symbol=' +
                  encodeURIComponent(sym) + '&interval=5m&range=1d';
        fetch(url)
            .then(function(r) {
                if (r.status === 429 && window.yahooProxyPace) window.yahooProxyPace.register429();
                return r.json();
            })
            .then(function(data) {
                if (activeRef !== activeCard || !candleSeries) return;
                var result = data && data.chart && data.chart.result && data.chart.result[0];
                if (!result) return;
                var ts = result.timestamp;
                var q  = result.indicators.quote[0];
                if (!ts || !ts.length) return;
                var now2 = new Date();
                var todayDateStr = now2.getUTCFullYear() + '-' +
                    String(now2.getUTCMonth() + 1).padStart(2, '0') + '-' +
                    String(now2.getUTCDate()).padStart(2, '0');
                var open = null, high = null, low = null, close = null, vol = 0;
                for (var i = 0; i < ts.length; i++) {
                    if (q.open[i] == null || q.close[i] == null) continue;
                    var bd = new Date(ts[i] * 1000);
                    var bds = bd.getUTCFullYear() + '-' +
                        String(bd.getUTCMonth() + 1).padStart(2, '0') + '-' +
                        String(bd.getUTCDate()).padStart(2, '0');
                    if (bds !== todayDateStr) continue;
                    if (open === null) open = q.open[i];
                    high  = high === null ? q.high[i]  : Math.max(high,  q.high[i]);
                    low   = low  === null ? q.low[i]   : Math.min(low,   q.low[i]);
                    close = q.close[i];
                    vol  += (q.volume[i] || 0);
                }
                if (open === null) return;
                if (!todayTs) {
                    var now = new Date();
                    todayTs = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 1000;
                }
                candleSeries.update({ time: todayTs, open: open, high: high, low: low, close: close, volume: vol });
            })
            .catch(function() {});
    }

    function showPopup(card) {
        if (!ready) init();
        if (!popup) return;

        var yfSym  = card.getAttribute('data-yf-sym');
        var label  = card.getAttribute('data-tv-label');
        var name   = card.getAttribute('data-tv-name');
        var price  = card.getAttribute('data-tv-price');
        var chgAbs = card.getAttribute('data-tv-chgabs');
        var chgPct = card.getAttribute('data-tv-chgpct');
        var dir    = card.getAttribute('data-tv-dir');

        if (!yfSym) return;

        activeYfSym = yfSym;
        popupSym.textContent  = label;
        popupChg.textContent   = chgAbs + ' ' + chgPct;
        popupChg.style.color   = dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#484f58';
        if (popupName) popupName.textContent = name || '';

        clearInterval(liveRefreshTimer);
        liveRefreshTimer = setInterval(function() {
            if (!activeCard) return;
            var liveChgAbs = activeCard.getAttribute('data-tv-chgabs');
            var liveChgPct = activeCard.getAttribute('data-tv-chgpct');
            var liveDir    = activeCard.getAttribute('data-tv-dir');
            if (liveChgAbs != null && liveChgPct != null) {
                popupChg.textContent = liveChgAbs + ' ' + liveChgPct;
                popupChg.style.color = liveDir === 'up' ? '#3fb950' : liveDir === 'down' ? '#f85149' : '#484f58';
            }
        }, 5000);
        if (popupIndName) { popupIndName.textContent = ''; }
        if (popupIndSep)  { popupIndSep.style.display = 'none'; }
        if (popupIndRank)   popupIndRank.style.display   = 'none';
        if (popupRsBadge)   popupRsBadge.style.display   = 'none';
        if (popup3mrsBadge) popup3mrsBadge.style.display = 'none';

        positionPopup(card);
        popup.classList.add('visible');
        popupChart.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:220px;color:#484f58;font-size:12px;">Loading…</div>';

        // Skip rather than pile onto an active shared cooldown; show the
        // existing "unavailable" state immediately instead of leaving the
        // popup stuck on "Loading…" until it times out on its own.
        if (window.yahooProxyPace && Date.now() < window.yahooProxyPace.cooldownUntil()) {
            popupChart.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:220px;color:#484f58;font-size:12px;">Chart unavailable</div>';
            return;
        }

        var url = 'https://yahoo-proxy.jay69k.workers.dev?symbol=' +
                  encodeURIComponent(yfSym) + '&interval=1d&range=6mo';

        fetch(url)
            .then(function(r) {
                if (r.status === 429 && window.yahooProxyPace) window.yahooProxyPace.register429();
                return r.json();
            })
            .then(function(data) {
                if (card !== activeCard) return; // user moved away
                var result = data && data.chart && data.chart.result && data.chart.result[0];
                if (!result) throw new Error('no data');

                var ts     = result.timestamp;
                var q      = result.indicators.quote[0];
                var ohlcv  = [];
                for (var i = 0; i < ts.length; i++) {
                    if (q.open[i] == null || q.close[i] == null) continue;
                    ohlcv.push({
                        time:   ts[i],
                        open:   q.open[i],
                        high:   q.high[i],
                        low:    q.low[i],
                        close:  q.close[i],
                        volume: q.volume[i] || 0,
                    });
                }
                var now = new Date();
                var todayStr = now.getUTCFullYear() + '-' +
                    String(now.getUTCMonth() + 1).padStart(2, '0') + '-' +
                    String(now.getUTCDate()).padStart(2, '0');
                var capturedTodayTs = null;
                if (ohlcv.length) {
                    var last = ohlcv[ohlcv.length - 1];
                    var lastD = new Date(last.time * 1000);
                    var lastDs = lastD.getUTCFullYear() + '-' +
                        String(lastD.getUTCMonth() + 1).padStart(2, '0') + '-' +
                        String(lastD.getUTCDate()).padStart(2, '0');
                    if (lastDs === todayStr) {
                        capturedTodayTs = ohlcv.pop().time;
                    }
                }
                renderChart(ohlcv);
                appendTodayCandle(yfSym, card, capturedTodayTs);
            })
            .catch(function() {
                if (popupChart) popupChart.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:220px;color:#484f58;font-size:12px;">Chart unavailable</div>';
            });
    }

    function hidePopup() {
        if (!popup) return;
        clearInterval(liveRefreshTimer);
        liveRefreshTimer = null;
        popup.classList.remove('visible');
        destroyChart();
        activeCard = null;
    }

    // Track mouse position and hide only when truly outside both card and popup
    function startMouseTracking() {
        document.addEventListener('mousemove', function (e) {
            if (!popup || !popup.classList.contains('visible') || !activeCard) return;
            // Don't hide while the watchlist/alert picker is open — its backdrop
            // sits on top of the popup and would otherwise trigger the hide timer
            var picker = document.getElementById('wl-picker');
            if (picker && picker.style.display !== 'none') return;
            var el       = document.elementFromPoint(e.clientX, e.clientY);
            var overCard = activeCard.contains(el) || el === activeCard;
            var overPop  = popup.contains(el)      || el === popup;
            if (overCard || overPop) {
                clearTimeout(hideTimer);
            } else {
                clearTimeout(hideTimer);
                hideTimer = setTimeout(hidePopup, 120);
            }
        });
    }

    function bindCard(card) {
        card.addEventListener('mouseover', function (e) {
            if (card.contains(e.relatedTarget)) return;
            if (card === activeCard) return;
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            activeCard = card;
            hoverTimer = setTimeout(function () { showPopup(card); }, 200);
        });
    }

    function showTickerPopup(el, ticker) {
        if (!ready) init();
        if (!popup) return;

        var _td = (typeof window._getTickerPopupData === 'function') ? window._getTickerPopupData(ticker) : {};
        var name     = _td.name || '';
        var industry = _td.industry || '';
        var price    = _td.price != null ? _td.price : null;
        var prevClose = _td.prevClose != null ? _td.prevClose : null;

        var chgAbs = (price != null && prevClose != null) ? price - prevClose : null;
        var chgPct = (chgAbs != null && prevClose > 0) ? (chgAbs / prevClose) * 100 : null;
        var dir    = chgPct == null ? 'flat' : chgPct > 0 ? 'up' : chgPct < 0 ? 'down' : 'flat';

        popupSym.textContent = ticker;
        activeYfSym = null;
        if (popupName) popupName.textContent = name || '';

        // Industry name in header
        if (popupIndName) { popupIndName.textContent = industry; }
        if (popupIndSep)  { popupIndSep.style.display = industry ? '' : 'none'; }

        var chgAbsStr = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '';
        var chgPctStr = chgPct != null ? ' (' + (chgPct >= 0 ? '+' : '') + chgPct.toFixed(2) + '%)' : '';
        popupChg.textContent = chgAbsStr + chgPctStr;
        popupChg.style.color = dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#484f58';

        clearInterval(liveRefreshTimer);
        liveRefreshTimer = setInterval(function() {
            var ltd = (typeof window._getTickerPopupData === 'function') ? window._getTickerPopupData(ticker) : {};
            var livePrice     = ltd.price != null ? ltd.price : null;
            var livePrevClose = ltd.prevClose != null ? ltd.prevClose : null;
            var liveChgAbs = (livePrice != null && livePrevClose != null) ? livePrice - livePrevClose : null;
            var liveChgPct = (liveChgAbs != null && livePrevClose > 0) ? (liveChgAbs / livePrevClose) * 100 : null;
            var liveDir    = liveChgPct == null ? 'flat' : liveChgPct > 0 ? 'up' : liveChgPct < 0 ? 'down' : 'flat';
            var liveAbsStr = liveChgAbs != null ? (liveChgAbs >= 0 ? '+' : '') + liveChgAbs.toFixed(2) : '';
            var livePctStr = liveChgPct != null ? ' (' + (liveChgPct >= 0 ? '+' : '') + liveChgPct.toFixed(2) + '%)' : '';
            if (liveAbsStr || livePctStr) {
                popupChg.textContent = liveAbsStr + livePctStr;
                popupChg.style.color = liveDir === 'up' ? '#3fb950' : liveDir === 'down' ? '#f85149' : '#484f58';
            }
        }, 5000);

        // Industry rank
        if (popupIndRank) {
            if (_td.indRank != null && _td.indTotal != null) {
                var rankColor = _td.indPct != null ? (_td.indPct >= 75 ? '#3fb950' : _td.indPct >= 40 ? '#e3852b' : '#f85149') : '#6e7681';
                popupIndRank.textContent   = '(' + _td.indRank + '/' + _td.indTotal + ')';
                popupIndRank.style.color   = rankColor;
                popupIndRank.style.display = '';
            } else {
                popupIndRank.style.display = 'none';
            }
        }

        // RS badge
        if (popupRsBadge) {
            var rsVal = _td.rs != null ? Math.round(_td.rs) : null;
            if (rsVal != null) {
                var rsCls = rsVal >= 75 ? 'rs-high' : rsVal >= 40 ? 'rs-mid' : 'rs-low';
                popupRsBadge.className   = 'chart-rs-badge ' + rsCls;
                popupRsBadge.textContent = 'RS ' + rsVal;
                popupRsBadge.style.display = '';
            } else {
                popupRsBadge.style.display = 'none';
            }
        }
        // 3MRS badge
        if (popup3mrsBadge) {
            var rs3mVal = _td.rs3m != null ? Math.round(_td.rs3m) : null;
            if (rs3mVal != null) {
                var rs3mCls = rs3mVal >= 75 ? 'rs-high' : rs3mVal >= 40 ? 'rs-mid' : 'rs-low';
                popup3mrsBadge.className   = 'chart-rs-badge ' + rs3mCls;
                popup3mrsBadge.textContent = rs3mVal;
                popup3mrsBadge.style.display = '';
            } else {
                popup3mrsBadge.style.display = 'none';
            }
        }

        positionPopup(el);
        popup.classList.add('visible');
        popupChart.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:220px;color:#484f58;font-size:12px;">Loading…</div>';

        if (window.yahooProxyPace && Date.now() < window.yahooProxyPace.cooldownUntil()) {
            popupChart.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:220px;color:#484f58;font-size:12px;">Chart unavailable</div>';
            return;
        }

        var url = 'https://yahoo-proxy.jay69k.workers.dev?symbol=' +
                  encodeURIComponent(ticker) + '&interval=1d&range=6mo';

        fetch(url)
            .then(function(r) {
                if (r.status === 429 && window.yahooProxyPace) window.yahooProxyPace.register429();
                return r.json();
            })
            .then(function(data) {
                if (el !== activeCard) return;
                var result = data && data.chart && data.chart.result && data.chart.result[0];
                if (!result) throw new Error('no data');
                var meta = result.meta;
                var resolvedName = (meta && (meta.shortName || meta.longName)) || '';
                if (resolvedName && popupName) popupName.textContent = resolvedName;
                var ts    = result.timestamp;
                var q     = result.indicators.quote[0];
                var ohlcv = [];
                for (var i = 0; i < ts.length; i++) {
                    if (q.open[i] == null || q.close[i] == null) continue;
                    ohlcv.push({
                        time:   ts[i],
                        open:   q.open[i],
                        high:   q.high[i],
                        low:    q.low[i],
                        close:  q.close[i],
                        volume: q.volume[i] || 0,
                    });
                }
                var now = new Date();
                var todayStr = now.getUTCFullYear() + '-' +
                    String(now.getUTCMonth() + 1).padStart(2, '0') + '-' +
                    String(now.getUTCDate()).padStart(2, '0');
                var capturedTodayTs = null;
                if (ohlcv.length) {
                    var last = ohlcv[ohlcv.length - 1];
                    var lastD = new Date(last.time * 1000);
                    var lastDs = lastD.getUTCFullYear() + '-' +
                        String(lastD.getUTCMonth() + 1).padStart(2, '0') + '-' +
                        String(lastD.getUTCDate()).padStart(2, '0');
                    if (lastDs === todayStr) {
                        capturedTodayTs = ohlcv.pop().time;
                    }
                }
                renderChart(ohlcv);
                appendTodayCandle(ticker, el, capturedTodayTs);
            })
            .catch(function() {
                if (popupChart) popupChart.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:220px;color:#484f58;font-size:12px;">Chart unavailable</div>';
            });
    }

    function bindTicker(el, ticker) {
        el.addEventListener('mouseenter', function() {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            activeCard = el;
            hoverTimer = setTimeout(function() { showTickerPopup(el, ticker); }, 200);
        });
        el.addEventListener('mouseleave', function() {
            clearTimeout(hoverTimer);
            hideTimer = setTimeout(hidePopup, 120);
        });
    }

    document.addEventListener('DOMContentLoaded', function () { init(); startMouseTracking(); });
    return { bindCard: bindCard, hide: hidePopup, bindTicker: bindTicker };
})();

function mmBindHover(card) { _mmPopup.bindCard(card); }

// Bind hover chart popup to all ticker elements in a container.
// selector: CSS selector for ticker elements (e.g. '.ticker-badge')
// tickerFn: optional function(el) → ticker string; defaults to el.textContent.trim()
window.tickerHoverBind = function(container, selector, tickerFn) {
    if (!container) return;
    container.querySelectorAll(selector).forEach(function(el) {
        var ticker = tickerFn ? tickerFn(el) : el.textContent.trim();
        if (ticker) _mmPopup.bindTicker(el, ticker);
    });
};
// ── END Macro card hover chart popup ─────────────────────────────────────

