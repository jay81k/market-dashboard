    // ── Market Overview ───────────────────────────────────────────────────

    var MARKET_INDEXES = [
        { id: 'GSPC', symbol: '^GSPC', label: 'SPX',  name: 'S&P 500',      futures: 'ES=F' },
        { id: 'IXIC', symbol: '^IXIC', label: 'NAS',  name: 'Nasdaq',        futures: 'NQ=F' },
        { id: 'DJI',  symbol: '^DJI',  label: 'DJI',  name: 'Dow Jones',     futures: 'YM=F' },
        { id: 'RUT',  symbol: '^RUT',  label: 'RUT',  name: 'Russell 2000',  futures: 'RTY=F' },
    ];

    var marketTf       = '1d';
    var marketInterval = '5m';
    var marketTimer    = null;

    window.setMarketTf = function(btn) {
        marketTf       = btn.getAttribute('data-tf');
        marketInterval = btn.getAttribute('data-interval');
        document.querySelectorAll('.market-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b === btn);
        });
        marketFetchAll();
        marketFetchMacro();
    };

    function marketFetchOne(symbol) {
        var url = WL_PROXY + '?symbol=' + encodeURIComponent(symbol) +
                  '&interval=' + marketInterval + '&range=' + marketTf;
        return fetch(url)
            .then(function(r) { return r.ok ? r.json() : null; })
            .catch(function() { return null; });
    }

    function marketParseResult(data) {
        if (!data || !data.chart || !data.chart.result || !data.chart.result[0]) return null;
        var result     = data.chart.result[0];
        var meta       = result.meta || {};
        var timestamps = result.timestamp || [];
        var quote      = (result.indicators && result.indicators.quote && result.indicators.quote[0]) || {};
        var closes     = quote.close  || [];
        var opens      = quote.open   || [];
        var highs      = quote.high   || [];
        var lows       = quote.low    || [];
        var volumes    = quote.volume || [];

        var points = [], ts = [], ohlcv = [];
        for (var i = 0; i < timestamps.length; i++) {
            if (closes[i] != null) {
                points.push(closes[i]);
                ts.push(timestamps[i]);
                ohlcv.push({
                    time:   timestamps[i],
                    open:   opens[i]   != null ? opens[i]   : closes[i],
                    high:   highs[i]   != null ? highs[i]   : closes[i],
                    low:    lows[i]    != null ? lows[i]    : closes[i],
                    close:  closes[i],
                    volume: volumes[i] || 0,
                });
            }
        }

        // First real open value from the bars — most reliable source for intraday "from open"
        var firstOpen = null;
        for (var j = 0; j < opens.length; j++) {
            if (opens[j] != null) { firstOpen = opens[j]; break; }
        }

        return {
            price:      meta.regularMarketPrice                      || null,
            prevClose:  meta.chartPreviousClose || meta.previousClose || null,
            marketOpen: meta.regularMarketOpen || firstOpen          || null,
            points:     points,
            timestamps: ts,
            ohlcv:      ohlcv,
        };
    }

    function marketFmtPrice(v) {
        if (v == null) return '—';
        return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function marketRenderIndexChart(container, parsed, direction) {
        if (!container || !parsed || !parsed.ohlcv || parsed.ohlcv.length < 2) return;
        if (typeof LightweightCharts === 'undefined') return;

        // Destroy any previous chart on this container
        if (container._lwChart) {
            try { container._lwChart.remove(); } catch(e) {}
            container._lwChart = null;
        }
        container.innerHTML = '';
        container.style.position = 'relative';

        var ohlcv = parsed.ohlcv;

        var lwChart = LightweightCharts.createChart(container, {
            width:  container.clientWidth  || 400,
            height: container.clientHeight || 230,
            layout: { 
                background: { color: '#0d1117' }, 
                textColor: '#6e7681',
                padding: { top: 0, bottom: 0, left: 0, right: 0 }
            },
            grid: { 
                vertLines: { visible: false }, 
                horzLines: { color: '#21262d' } 
            },
            crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
            rightPriceScale: { 
                borderVisible: false,
                textColor: '#6e7681' 
            },
            timeScale: { 
                visible: false
            },
            handleScroll: false,
            handleScale:  false,
        });
        container._lwChart = lwChart;

        // Candlestick series
        var candleSeries = lwChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor:               '#3fb950',
            downColor:             '#f85149',
            borderVisible:         false,
            wickUpColor:           '#3fb950',
            wickDownColor:         '#f85149',
            priceLineVisible:      false,
            lastValueVisible:      true,
        });
        candleSeries.setData(ohlcv);

        // Volume histogram
        var volSeries = lwChart.addSeries(LightweightCharts.HistogramSeries, {
            priceFormat:  { type: 'volume' },
            priceScaleId: 'volume',
        });
        lwChart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.82, bottom: 0 },
            visible: false,
        });
        volSeries.setData(ohlcv.map(function(d) {
            return {
                time:  d.time,
                value: d.volume,
                color: d.close >= d.open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)',
            };
        }));

        lwChart.timeScale().fitContent();

        // OHLC crosshair readout — hidden until hover
        var ohlcLegend = document.createElement('div');
        ohlcLegend.style.cssText = [
            'position:absolute', 'top:6px', 'left:6px', 'z-index:20',
            'font-size:10px', 'font-weight:600', 'font-variant-numeric:tabular-nums',
            'color:#8b949e', 'pointer-events:none', 'line-height:1.5',
            'background:rgba(13,17,23,0.88)', 'padding:2px 6px', 'border-radius:3px',
            'display:none',
        ].join(';');
        container.appendChild(ohlcLegend);

        function fmtP(v) { return v != null ? v.toFixed(2) : '—'; }
        function fmtV(v) {
            if (v == null) return '—';
            return v >= 1e9 ? (v / 1e9).toFixed(1) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : v.toFixed(0);
        }

        lwChart.subscribeCrosshairMove(function(param) {
            if (!param.time || !param.seriesData || !param.seriesData.size) {
                ohlcLegend.style.display = 'none';
                return;
            }
            var d = param.seriesData.get(candleSeries);
            if (!d) { ohlcLegend.style.display = 'none'; return; }
            ohlcLegend.style.display = '';
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

    function marketRenderCard(idx, parsed, futuresParsed) {
        var card = document.getElementById('mc-' + idx.id);
        if (!card) return;

        card.style.padding  = '0';
        card.style.overflow = 'hidden';

        if (!parsed) {
            card.className = 'market-index-card';
            card.innerHTML =
                '<div style="height:230px;position:relative;background:#0d1117;">' +
                    '<div style="position:absolute;top:8px;left:10px;z-index:5;pointer-events:none;">' +
                        '<div style="font-size:12px;font-weight:700;color:#484f58;letter-spacing:0.06em;">' + idx.label + '</div>' +
                        '<div style="font-size:10px;color:#484f58;">' + idx.name + '</div>' +
                    '</div>' +
                    '<div style="position:absolute;bottom:10px;left:10px;font-size:18px;font-weight:700;color:#484f58;font-variant-numeric:tabular-nums;z-index:5;">—</div>' +
                '</div>';
            return;
        }

        var price     = parsed.price;
        var prevClose = parsed.prevClose;
        var chgAbs    = (price != null && prevClose != null) ? price - prevClose : null;
        var chgPct    = (chgAbs != null && prevClose > 0) ? (chgAbs / prevClose) * 100 : null;
        var direction = chgPct == null ? 'flat' : chgPct > 0 ? 'up' : chgPct < 0 ? 'down' : 'flat';

        var marketOpen   = parsed.marketOpen;
        var fromOpenPct  = (price != null && marketOpen != null && marketOpen > 0) ? ((price - marketOpen) / marketOpen) * 100 : null;
        var fromOpenDir  = fromOpenPct == null ? 'flat' : fromOpenPct > 0 ? 'up' : fromOpenPct < 0 ? 'down' : 'flat';
        var fromOpenStr  = fromOpenPct != null ? (fromOpenPct >= 0 ? '▲ +' : '▼ ') + fromOpenPct.toFixed(2) + '%' : null;

        var priceStr  = marketFmtPrice(price);
        var chgAbsStr = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '—';
        var chgPctStr = chgPct != null ? (chgPct >= 0 ? '+' : '') + chgPct.toFixed(2) + '%' : '—';

        var isOpen   = wlIsMarketOpen();
        var is1D     = marketTf === '1d';

        // Futures pill — 1D only, outside market hours
        var futuresPillHtml = '';
        if (is1D && !isOpen && futuresParsed && futuresParsed.price != null && futuresParsed.prevClose != null) {
            var fChgPct = ((futuresParsed.price - futuresParsed.prevClose) / futuresParsed.prevClose) * 100;
            var fDir    = fChgPct > 0 ? 'up' : fChgPct < 0 ? 'down' : 'flat';
            var fStr    = (fChgPct >= 0 ? '▲ +' : '▼ ') + fChgPct.toFixed(2) + '%';
            futuresPillHtml =
                '<div class="mic-futures-pill ' + fDir + '" style="display:inline-flex;">' +
                    '<span class="mic-futures-sym">' + idx.futures + '</span>' +
                    '<span class="mic-futures-val">' + fStr + '</span>' +
                '</div>';
        }

        var chartContainerId = 'mic-lwchart-' + idx.id;
        card.className = 'market-index-card ' + direction;

        // Chart wrap is the only child — fills the entire card
        card.innerHTML =
            '<div class="mic-chart-wrap" id="' + chartContainerId + '" style="height:230px;margin:0;"></div>';

        var chartContainer = document.getElementById(chartContainerId);

        // Render chart first (clears container internally, then adds canvas + ohlcLegend)
        marketRenderIndexChart(chartContainer, parsed, direction);

        // ── Overlay: top-left — name + futures pill on row 1, price/change/from-open below ──
        var dirColor = direction === 'up' ? '#3fb950' : direction === 'down' ? '#f85149' : '#8b949e';
        var topLeft = document.createElement('div');
        // right:75px keeps us clear of the price axis
        topLeft.style.cssText = 'position:absolute;top:8px;left:10px;right:75px;z-index:5;pointer-events:none;';

        var foColor  = fromOpenDir === 'up' ? '#3fb950' : fromOpenDir === 'down' ? '#f85149' : '#8b949e';
        var fromOpenRowHtml = (is1D && fromOpenStr != null)
            ? '<div style="font-size:10px;color:#8b949e;margin-top:2px;">from open <span style="color:' + foColor + ';">' + fromOpenStr + '</span></div>'
            : '';

        topLeft.innerHTML =
            // Row 1: index name + futures pill
            '<div style="display:flex;align-items:center;gap:6px;line-height:1.3;">' +
                '<span style="font-size:12px;font-weight:700;color:#e6edf3;">' + idx.name + '</span>' +
                futuresPillHtml +
            '</div>' +
            // Row 2: price + change
            '<div style="display:flex;align-items:baseline;gap:5px;margin-top:4px;">' +
                '<span style="font-size:18px;font-weight:700;color:#e6edf3;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;line-height:1;">' + priceStr + '</span>' +
                '<span style="font-size:11px;font-weight:600;color:' + dirColor + ';font-variant-numeric:tabular-nums;">' + chgAbsStr + ' (' + chgPctStr + ')</span>' +
            '</div>' +
            // Row 3: from open
            fromOpenRowHtml;

        chartContainer.appendChild(topLeft);
    }

    function marketFetchAll() {
        // Reset to skeleton while loading
        MARKET_INDEXES.forEach(function(idx) {
            var card = document.getElementById('mc-' + idx.id);
            if (card) { card.className = 'market-index-card skeleton'; card.innerHTML = ''; }
        });
        document.getElementById('market-updated').textContent = '';

        Promise.all(MARKET_INDEXES.map(function(idx) {
            return Promise.all([
                marketFetchOne(idx.symbol).then(function(data) { return marketParseResult(data); }),
                marketFetchOne(idx.futures).then(function(data) { return marketParseResult(data); }),
            ]);
        })).then(function(results) {
            MARKET_INDEXES.forEach(function(idx, i) {
                marketRenderCard(idx, results[i][0], results[i][1]);
            });
            renderMarketBreadth();
            renderMarketHL();
            renderMarketMA();
            renderMarketMovers();
            renderSectorPerf();
            renderIndBreadth();
            renderRSDist();
            marketFetchMacro();
            var now = new Date();
            var isOpen = wlIsMarketOpen();
            document.getElementById('market-updated').textContent =
                'Updated ' + now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) +
                (isOpen ? '' : ' · Market closed');

            // Only auto-refresh during market hours — set timer once, not on every fetch
            if (!marketTimer) {
                if (isOpen) {
                    marketTimer = setInterval(function() {
                        if (currentView !== 'market') { clearInterval(marketTimer); marketTimer = null; return; }
                        if (!wlIsMarketOpen()) { clearInterval(marketTimer); marketTimer = null; return; }
                        marketFetchAll();
                    }, 5 * 60 * 1000);
                }
            }
        });
    }

    function marketStopTimer() {
        if (marketTimer) { clearInterval(marketTimer); marketTimer = null; }
    }

    var MACRO_TICKERS = [
        { id: 'OIL',  symbol: 'CL=F',    label: 'OIL',    name: 'Crude Oil WTI' },
        { id: 'GLD',  symbol: 'GC=F',    label: 'GOLD',   name: 'Gold Futures' },
        { id: 'SLV',  symbol: 'SI=F',    label: 'SILVER', name: 'Silver Futures' },
        { id: 'CPR',  symbol: 'HG=F',    label: 'COPPER', name: 'Copper Futures' },
        { id: 'VIX',  symbol: '^VIX',    label: 'VIX',    name: 'Volatility Index' },
        { id: 'TNX',  symbol: '^TNX',    label: '10YR',   name: '10-Yr Yield' },
        { id: 'ETH',  symbol: 'ETH-USD', label: 'ETH',    name: 'Ethereum' },
        { id: 'XRP',  symbol: 'XRP-USD', label: 'XRP',    name: 'XRP' },
        { id: 'SOL',  symbol: 'SOL-USD', label: 'SOL',    name: 'Solana' },
        { id: 'HYPE', symbol: 'HYPE32196-USD', label: 'HYPE',   name: 'Hyperliquid' },
        { id: 'BTC',  symbol: 'BTC-USD', label: 'BTC',    name: 'Bitcoin' },
    ];

    function marketFetchMacro() {
        // Macro cards always use daily interval — 1D locks to range=2d,
        // other timeframes use the active range with daily candles
        var range    = (marketTf === '1d') ? '2d'      : marketTf;
        var interval = (marketTf === '1d') ? '1d'      : marketInterval;
        Promise.all(MACRO_TICKERS.map(function(t) {
            return fetch(WL_PROXY + '?symbol=' + encodeURIComponent(t.symbol) + '&interval=' + interval + '&range=' + range)
                .then(function(r) { return r.ok ? r.json() : null; })
                .catch(function() { return null; });
        })).then(function(results) {
            MACRO_TICKERS.forEach(function(t, i) {
                var el = document.getElementById('mm-' + t.id);
                if (!el) return;

                var data   = results[i];
                var result = data && data.chart && data.chart.result && data.chart.result[0];
                if (!result) {
                    el.className = 'market-macro-card flat';
                    el.innerHTML = '<div class="mm-label">' + t.label + '</div><div class="mm-name">' + t.name + '</div><div class="mm-price">—</div>';
                    return;
                }

                var meta      = result.meta || {};
                var price     = meta.regularMarketPrice;
                var prevClose = meta.previousClose || meta.chartPreviousClose;
                var dayHigh   = meta.regularMarketDayHigh;
                var dayLow    = meta.regularMarketDayLow;
                var chgAbs    = (price != null && prevClose != null) ? price - prevClose : null;
                var chgPct    = (chgAbs != null && prevClose > 0) ? (chgAbs / prevClose) * 100 : null;
                var dir       = chgPct == null ? 'flat' : chgPct > 0 ? 'up' : chgPct < 0 ? 'down' : 'flat';

                var isCrypto  = ['BTC','ETH','XRP','SOL','HYPE'].indexOf(t.id) !== -1;
                var priceStr  = price != null ? price.toLocaleString('en-US', { minimumFractionDigits: isCrypto ? 0 : 2, maximumFractionDigits: isCrypto ? (price >= 100 ? 0 : 2) : 2 }) : '—';
                var chgAbsStr = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '—';
                var chgPctStr = chgPct != null ? (chgPct >= 0 ? '+' : '') + chgPct.toFixed(2) + '%' : '—';
                var hiStr     = dayHigh != null ? dayHigh.toFixed(2) : '—';
                var loStr     = dayLow  != null ? dayLow.toFixed(2)  : '—';

                el.className = 'market-macro-card ' + dir;
                el.setAttribute('data-tv-id',    t.id);
                el.setAttribute('data-yf-sym',   t.symbol);
                el.setAttribute('data-tv-label', t.label);
                el.setAttribute('data-tv-name',  t.name);
                el.setAttribute('data-tv-price', priceStr);
                el.setAttribute('data-tv-chgabs', chgAbsStr);
                el.setAttribute('data-tv-chgpct', '(' + chgPctStr + ')');
                el.setAttribute('data-tv-dir',   dir);
                el.innerHTML =
                    '<div class="mm-label">' + t.label + '</div>' +
                    '<div class="mm-name">'  + t.name  + '</div>' +
                    '<div class="mm-price">' + priceStr + '</div>' +
                    '<div class="mm-chg">' +
                        '<span class="mm-chg-abs ' + dir + '">' + chgAbsStr + '</span>' +
                        '<span class="mm-chg-pct ' + dir + '">(' + chgPctStr + ')</span>' +
                    '</div>' +
                    '<div class="mm-hl">' +
                        '<div class="mm-hl-item"><span>H </span>' + hiStr + '</div>' +
                        '<div class="mm-hl-item"><span>L </span>' + loStr + '</div>' +
                    '</div>';

                // Attach hover listeners directly now that card is rendered
                if (typeof mmBindHover === 'function') mmBindHover(el);
            });
        });
    }

    function renderMarketBreadth() {
        var wrap = document.getElementById('market-ad-wrap');
        if (!wrap || !snapshot || !snapshot.by_industry) return;

        var adv = 0, dec = 0, unch = 0;
        Object.values(snapshot.by_industry).forEach(function(rows) {
            rows.forEach(function(r) {
                if (r.daily == null) return;
                if (r.daily > 0)      adv++;
                else if (r.daily < 0) dec++;
                else                  unch++;
            });
        });

        var total   = adv + dec + unch;
        if (!total) { wrap.innerHTML = '<div style="color:#484f58;font-size:0.858em;">No data.</div>'; return; }

        var advPct  = (adv  / total * 100).toFixed(1);
        var decPct  = (dec  / total * 100).toFixed(1);
        var unPct   = (unch / total * 100).toFixed(1);
        var ratio   = dec > 0 ? (adv / dec).toFixed(2) : '—';
        var net     = adv - dec;
        var netCl   = net > 0 ? 'up' : net < 0 ? 'down' : '';
        var netStr  = (net >= 0 ? '+' : '') + net;

        // Bar widths — proportional, unch gets minimum visual space if > 0
        var advW  = (adv  / total * 100).toFixed(2) + '%';
        var decW  = (dec  / total * 100).toFixed(2) + '%';
        var unW   = (unch / total * 100).toFixed(2) + '%';

        wrap.innerHTML =
            '<div class="market-ad-row">' +
                '<div class="market-ad-side-adv">' +
                    '<div class="market-ad-label adv">Advancing</div>' +
                    '<div class="market-ad-count adv">' + adv.toLocaleString() + ' <span class="market-ad-pct">(' + advPct + '%)</span></div>' +
                '</div>' +
                '<div class="market-ad-bar-wrap">' +
                    '<div class="market-ad-bar-adv"  style="width:' + advW + '"></div>' +
                    (unch ? '<div class="market-ad-bar-unch" style="width:' + unW + '"></div>' : '') +
                    '<div class="market-ad-bar-dec"  style="width:' + decW + '"></div>' +
                '</div>' +
                '<div class="market-ad-side-dec">' +
                    '<div class="market-ad-label dec">Declining</div>' +
                    '<div class="market-ad-count dec">' + dec.toLocaleString() + ' <span class="market-ad-pct">(' + decPct + '%)</span></div>' +
                '</div>' +
            '</div>' +
            '<div class="market-ad-ratio">' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">A/D Ratio</span><span class="market-ad-stat-val ' + netCl + '">' + ratio + '</span></div>' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">Net</span><span class="market-ad-stat-val ' + netCl + '">' + netStr + '</span></div>' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">Unchanged</span><span class="market-ad-stat-val">' + unch + '</span></div>' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">Total</span><span class="market-ad-stat-val">' + total.toLocaleString() + '</span></div>' +
            '</div>';
    }

    function renderMarketHL() {
        var wrap = document.getElementById('market-hl-wrap');
        if (!wrap || !snapshot || !snapshot.by_industry) return;

        var nh = 0, nl = 0, total = 0;
        Object.values(snapshot.by_industry).forEach(function(rows) {
            rows.forEach(function(r) {
                total++;
                if (r.new_52wk_high === true) nh++;
                if (r.new_52wk_low  === true) nl++;
            });
        });

        if (!total) { wrap.innerHTML = '<div style="color:#484f58;font-size:0.858em;">No data.</div>'; return; }

        var nhPct  = (nh / total * 100).toFixed(1);
        var nlPct  = (nl / total * 100).toFixed(1);
        var net    = nh - nl;
        var netCl  = net > 0 ? 'up' : net < 0 ? 'down' : '';
        var netStr = (net >= 0 ? '+' : '') + net;
        var ratio  = nl > 0 ? (nh / nl).toFixed(2) : nh > 0 ? '∞' : '—';

        // Bar: NH green on left, rest grey, NL red on right
        var nhW = (nh / total * 100).toFixed(2) + '%';
        var nlW = (nl / total * 100).toFixed(2) + '%';

        wrap.innerHTML =
            '<div class="market-ad-row">' +
                '<div class="market-ad-side-adv">' +
                    '<div class="market-ad-label adv">New Highs</div>' +
                    '<div class="market-ad-count adv">' + nh.toLocaleString() + ' <span class="market-ad-pct">(' + nhPct + '%)</span></div>' +
                '</div>' +
                '<div class="market-ad-bar-wrap">' +
                    '<div class="market-ad-bar-adv" style="width:' + nhW + '"></div>' +
                    '<div class="market-ad-bar-unch" style="flex:1"></div>' +
                    '<div class="market-ad-bar-dec" style="width:' + nlW + '"></div>' +
                '</div>' +
                '<div class="market-ad-side-dec">' +
                    '<div class="market-ad-label dec">New Lows</div>' +
                    '<div class="market-ad-count dec">' + nl.toLocaleString() + ' <span class="market-ad-pct">(' + nlPct + '%)</span></div>' +
                '</div>' +
            '</div>' +
            '<div class="market-ad-ratio">' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">H/L Ratio</span><span class="market-ad-stat-val ' + netCl + '">' + ratio + '</span></div>' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">Net</span><span class="market-ad-stat-val ' + netCl + '">' + netStr + '</span></div>' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">Neither</span><span class="market-ad-stat-val">' + (total - nh - nl).toLocaleString() + '</span></div>' +
                '<div class="market-ad-stat"><span class="market-ad-stat-label">Total</span><span class="market-ad-stat-val">' + total.toLocaleString() + '</span></div>' +
            '</div>';
    }

    function maZoneDir(pct)   { return pct >= 60 ? 'up' : pct <= 40 ? 'down' : 'flat'; }
    function maZoneLabel(pct) { return pct >= 60 ? 'Bullish' : pct <= 40 ? 'Bearish' : 'Neutral'; }
    function maBarColor(dir)  { return dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#484f58'; }
    function maFmtNet(n)      { return (n >= 0 ? '+' : '') + n.toLocaleString(); }

    function renderMarketMA() {
        if (!snapshot || !snapshot.by_industry) return;

        var sma5 = 0, ema21 = 0, sma50a = 0, sma50b = 0, sma200a = 0, sma200b = 0, total = 0;

        Object.values(snapshot.by_industry).forEach(function(rows) {
            rows.forEach(function(r) {
                var dm = r.dist_ma;
                if (!dm) return;
                total++;
                var above50 = dm.SMA50 != null && dm.SMA50 > 0;
                if (dm.SMA5  != null && dm.SMA5  > 0 && above50) sma5++;
                if (dm.EMA21 != null && dm.EMA21 > 0 && above50) ema21++;
                if (dm.SMA50  != null) { if (dm.SMA50  > 0) sma50a++;  else sma50b++;  }
                if (dm.SMA200 != null) { if (dm.SMA200 > 0) sma200a++; else sma200b++; }
            });
        });

        var tot = total || 1;

        // Simple split cards — 5 SMA and 21 EMA (above = filtered count, below = rest)
        var ids    = ['market-ma-5sma', 'market-ma-21ema'];
        var labels = ['5 SMA', '21 EMA'];
        var cnts   = [sma5, ema21];
        for (var i = 0; i < 2; i++) {
            var el = document.getElementById(ids[i]);
            if (!el) continue;
            var ab   = cnts[i], bl = tot - cnts[i];
            var stot = tot || 1;
            var pctA = (ab / stot * 100).toFixed(1);
            var pctB = (bl / stot * 100).toFixed(1);
            var ddir = maZoneDir(Math.round(ab / stot * 100));
            var net  = ab - bl;
            var ndir = net > 0 ? 'up' : net < 0 ? 'down' : 'flat';
            var barClr = ddir === 'up' ? '#3fb950' : '#484f58';
            el.style.alignItems = '';
            el.innerHTML =
                '<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:6px;">' +
                    '<div style="display:flex;flex-direction:column;gap:1px;">' +
                        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#3fb950;">Above</div>' +
                        '<div style="font-size:18px;font-weight:700;letter-spacing:-0.02em;line-height:1;font-variant-numeric:tabular-nums;color:#3fb950;">' + pctA + '<span style="font-size:11px;">%</span></div>' +
                        '<div style="font-size:10px;color:#3fb950;opacity:0.7;font-variant-numeric:tabular-nums;">' + ab.toLocaleString() + '</div>' +
                    '</div>' +
                    '<div class="ma-label" style="align-self:center;text-align:center;" data-tooltip="Also above 50 SMA">' + labels[i] + '</div>' +
                    '<div style="display:flex;flex-direction:column;gap:1px;align-items:flex-end;">' +
                        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#f85149;">Below</div>' +
                        '<div style="font-size:18px;font-weight:700;letter-spacing:-0.02em;line-height:1;font-variant-numeric:tabular-nums;color:#f85149;">' + pctB + '<span style="font-size:11px;">%</span></div>' +
                        '<div style="font-size:10px;color:#f85149;opacity:0.7;font-variant-numeric:tabular-nums;">' + bl.toLocaleString() + '</div>' +
                    '</div>' +
                '</div>' +
                '<div>' +
                    '<div style="height:5px;border-radius:3px;overflow:hidden;display:flex;">' +
                        '<div style="height:100%;width:' + pctA + '%;border-radius:3px 0 0 3px;background:' + barClr + ';"></div>' +
                        '<div style="height:100%;flex:1;border-radius:0 3px 3px 0;background:#f85149;"></div>' +
                    '</div>' +
                    '<div style="display:flex;justify-content:space-between;margin-top:8px;">' +
                        '<div style="display:flex;flex-direction:column;gap:1px;">' +
                            '<div style="font-size:9px;color:#484f58;text-transform:uppercase;letter-spacing:0.05em;">Net</div>' +
                            '<div style="font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;" class="ma-net ' + ndir + '">' + maFmtNet(net) + '</div>' +
                        '</div>' +
                        '<div style="display:flex;flex-direction:column;gap:1px;align-items:flex-end;">' +
                            '<div style="font-size:9px;color:#484f58;text-transform:uppercase;letter-spacing:0.05em;">Zone</div>' +
                            '<div style="font-size:11px;font-weight:700;" class="ma-zone ' + ddir + '">' + maZoneLabel(Math.round(ab / stot * 100)) + '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>';
        }

        // Split cards — 50 SMA and 200 SMA
        var splitIds    = ['market-ma-50sma',  'market-ma-200sma'];
        var splitLabels = ['50 SMA',           '200 SMA'];
        var splitAbove  = [sma50a,              sma200a];
        var splitBelow  = [sma50b,              sma200b];
        for (var j = 0; j < 2; j++) {
            var sel = document.getElementById(splitIds[j]);
            if (!sel) continue;
            var ab   = splitAbove[j], bl = splitBelow[j];
            var stot = (ab + bl) || 1;
            var pctA = (ab / stot * 100).toFixed(1);
            var pctB = (bl / stot * 100).toFixed(1);
            var ddir = maZoneDir(Math.round(ab / stot * 100));
            var net  = ab - bl;
            var ndir = net > 0 ? 'up' : net < 0 ? 'down' : 'flat';
            var barClr = ddir === 'up' ? '#3fb950' : '#484f58';
            sel.innerHTML =
                '<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:6px;">' +
                    '<div style="display:flex;flex-direction:column;gap:1px;">' +
                        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#3fb950;">Above</div>' +
                        '<div style="font-size:18px;font-weight:700;letter-spacing:-0.02em;line-height:1;font-variant-numeric:tabular-nums;color:#3fb950;">' + pctA + '<span style="font-size:11px;">%</span></div>' +
                        '<div style="font-size:10px;color:#3fb950;opacity:0.7;font-variant-numeric:tabular-nums;">' + ab.toLocaleString() + '</div>' +
                    '</div>' +
                    '<div class="ma-label" style="align-self:center;text-align:center;">' + splitLabels[j] + '</div>' +
                    '<div style="display:flex;flex-direction:column;gap:1px;align-items:flex-end;">' +
                        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#f85149;">Below</div>' +
                        '<div style="font-size:18px;font-weight:700;letter-spacing:-0.02em;line-height:1;font-variant-numeric:tabular-nums;color:#f85149;">' + pctB + '<span style="font-size:11px;">%</span></div>' +
                        '<div style="font-size:10px;color:#f85149;opacity:0.7;font-variant-numeric:tabular-nums;">' + bl.toLocaleString() + '</div>' +
                    '</div>' +
                '</div>' +
                '<div>' +
                    '<div style="height:5px;border-radius:3px;overflow:hidden;display:flex;">' +
                        '<div style="height:100%;width:' + pctA + '%;border-radius:3px 0 0 3px;background:' + barClr + ';"></div>' +
                        '<div style="height:100%;flex:1;border-radius:0 3px 3px 0;background:#f85149;"></div>' +
                    '</div>' +
                    '<div style="display:flex;justify-content:space-between;margin-top:8px;">' +
                        '<div style="display:flex;flex-direction:column;gap:1px;">' +
                            '<div style="font-size:9px;color:#484f58;text-transform:uppercase;letter-spacing:0.05em;">Net</div>' +
                            '<div style="font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;" class="ma-net ' + ndir + '">' + maFmtNet(net) + '</div>' +
                        '</div>' +
                        '<div style="display:flex;flex-direction:column;gap:1px;align-items:flex-end;">' +
                            '<div style="font-size:9px;color:#484f58;text-transform:uppercase;letter-spacing:0.05em;">Zone</div>' +
                            '<div style="font-size:11px;font-weight:700;" class="ma-zone ' + ddir + '">' + maZoneLabel(Math.round(ab / stot * 100)) + '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>';
        }
    }

    function fmtVol(v) {
        if (v == null) return '—';
        if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
        if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
        if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
        return v.toFixed(0);
    }

    function renderMarketMovers() {
        if (!snapshot || !snapshot.by_industry) return;

        var gainEl = document.getElementById('market-gainers');
        var lossEl = document.getElementById('market-losers');
        if (!gainEl || !lossEl) return;

        // Collect all stocks with a daily value
        var all = [];
        Object.values(snapshot.by_industry).forEach(function(rows) {
            rows.forEach(function(r) {
                if (r.daily != null && r.price != null) all.push(r);
            });
        });

        all.sort(function(a, b) { return b.daily - a.daily; });
        var gainers = all.slice(0, 8);
        var losers  = all.slice(-8).reverse();
        var top16   = gainers.concat(losers);

        // Render unusual volume immediately from snapshot — no fetch needed
        var rvolEl = document.getElementById('market-rvol');
        if (rvolEl) {
            var rvolStocks = all.filter(function(r) { return r.rel_vol != null; });
            rvolStocks.sort(function(a, b) { return b.rel_vol - a.rel_vol; });
            var topRvol = rvolStocks.slice(0, 8);
            rvolEl.innerHTML = '<div class="gl-header"><div class="gl-title" style="color:#58a6ff;">Unusual Volume</div><div class="gl-subtitle">vs 50-day avg</div></div>' +
                topRvol.map(function(r, i) {
                    var pctStr = (r.daily >= 0 ? '+' : '') + r.daily.toFixed(2) + '%';
                    var dirCl  = r.daily > 0 ? 'up' : r.daily < 0 ? 'down' : '';
                    return '<div class="gl-row gl-clickable" data-ticker="' + esc(r.ticker) + '" data-industry="' + esc(r.industry||'') + '">' +
                        '<div class="gl-rank">' + (i + 1) + '</div>' +
                        '<div class="gl-ticker">' + esc(r.ticker) + '</div>' +
                        '<div class="gl-name">' + esc(r.industry || '') + '</div>' +
                        '<div class="gl-pct ' + dirCl + '">' + pctStr + '</div>' +
                        '<div class="gl-vol" style="color:#58a6ff;font-weight:700;">' + r.rel_vol.toFixed(1) + 'x</div>' +
                    '</div>';
                }).join('');
            tickerHoverBind(rvolEl, '.gl-ticker');
        }

        function buildRows(stocks, dir) {
            return stocks.map(function(r, i) {
                var pctStr = (r.daily >= 0 ? '+' : '') + r.daily.toFixed(2) + '%';
                var vol    = r.rel_vol != null ? r.rel_vol.toFixed(1) + 'x' : '—';
                return '<div class="gl-row gl-clickable" data-ticker="' + esc(r.ticker) + '" data-industry="' + esc(r.industry||'') + '">' +
                    '<div class="gl-rank">' + (i + 1) + '</div>' +
                    '<div class="gl-ticker">' + esc(r.ticker) + '</div>' +
                    '<div class="gl-name">' + esc(r.industry || '') + '</div>' +
                    '<div class="gl-price">$' + r.price.toFixed(2) + '</div>' +
                    '<div class="gl-pct ' + dir + '">' + pctStr + '</div>' +
                    '<div class="gl-vol" style="color:#58a6ff;font-weight:700;">' + vol + '</div>' +
                '</div>';
            }).join('');
        }

        gainEl.innerHTML = '<div class="gl-header"><div class="gl-title up">Top Gainers</div><div class="gl-subtitle">daily % change</div></div>' + buildRows(gainers, 'up');
        lossEl.innerHTML = '<div class="gl-header"><div class="gl-title down">Top Losers</div><div class="gl-subtitle">daily % change</div></div>' + buildRows(losers, 'down');
        tickerHoverBind(gainEl, '.gl-ticker');
        tickerHoverBind(lossEl, '.gl-ticker');
    }

    // ── Sector Performance ────────────────────────────────────────────────
    function renderSectorPerf() {
        var el = document.getElementById('market-sector-perf');
        if (!el || !snapshot || !snapshot.industry_summary) return;

        // Aggregate avg_daily by sector
        var sectorMap = {};
        Object.values(snapshot.industry_summary).forEach(function(s) {
            if (!s.sector || s.avg_daily == null) return;
            if (!sectorMap[s.sector]) sectorMap[s.sector] = { sum: 0, count: 0 };
            sectorMap[s.sector].sum   += s.avg_daily;
            sectorMap[s.sector].count += 1;
        });

        var sectors = Object.keys(sectorMap).map(function(name) {
            return { name: name, avg: sectorMap[name].sum / sectorMap[name].count };
        });
        sectors.sort(function(a, b) { return b.avg - a.avg; });

        var maxAbs = Math.max.apply(null, sectors.map(function(s) { return Math.abs(s.avg); })) || 1;

        var rows = sectors.map(function(s) {
            var pct    = s.avg;
            var dir    = pct > 0 ? 'up' : pct < 0 ? 'down' : '';
            var clr    = pct > 0 ? '#3fb950' : '#f85149';
            var barPct = Math.abs(pct) / maxAbs * 48; // max 48% each side from center
            var barStyle, barLeft;
            if (pct >= 0) {
                barLeft  = '50%';
                barStyle = 'left:50%;width:' + barPct + '%;border-radius:0 3px 3px 0;';
            } else {
                barStyle = 'right:50%;width:' + barPct + '%;border-radius:3px 0 0 3px;';
            }
            var sign = pct >= 0 ? '+' : '';
            return '<div class="sp-row">' +
                '<div class="sp-name" onclick="goToSector(\'' + esc(s.name) + '\')">' + esc(s.name) + '</div>' +
                '<div class="sp-bar-wrap">' +
                    '<div class="sp-center"></div>' +
                    '<div class="sp-bar" style="background:' + clr + ';' + barStyle + '"></div>' +
                '</div>' +
                '<div class="sp-pct ' + dir + '">' + sign + pct.toFixed(2) + '%</div>' +
            '</div>';
        }).join('');

        el.innerHTML =
            '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;">' +
                '<div class="analytics-title">Sector Performance</div>' +
            '</div>' + rows;
    }

    // ── Industry Breadth ──────────────────────────────────────────────────
    function renderIndBreadth() {
        var el = document.getElementById('market-ind-breadth');
        if (!el || !snapshot || !snapshot.industry_summary) return;

        var adv = 0, dec = 0, unch = 0;
        var sectorMap = {};

        Object.entries(snapshot.industry_summary).forEach(function(entry) {
            var s = entry[1];
            if (!s.sector) return;
            if (!sectorMap[s.sector]) sectorMap[s.sector] = { adv: 0, dec: 0, unch: 0 };
            if (s.avg_daily == null)    { unch++; sectorMap[s.sector].unch++; }
            else if (s.avg_daily > 0)   { adv++;  sectorMap[s.sector].adv++;  }
            else                        { dec++;  sectorMap[s.sector].dec++;  }
        });

        var total = adv + dec + unch || 1;
        var advPct = (adv / total * 100).toFixed(1);
        var decPct = (dec / total * 100).toFixed(1);
        var net    = adv - dec;
        var ratio  = dec > 0 ? (adv / dec).toFixed(2) : adv > 0 ? '∞' : '—';
        var netDir = net > 0 ? 'up' : net < 0 ? 'down' : '';

        // Best/worst sector by adv ratio
        var sectorList = Object.keys(sectorMap).map(function(name) {
            var sd = sectorMap[name];
            var tot = sd.adv + sd.dec + sd.unch || 1;
            return { name: name, adv: sd.adv, dec: sd.dec, ratio: sd.adv / tot };
        });
        sectorList.sort(function(a, b) { return b.ratio - a.ratio; });

        var secRows = sectorList.map(function(s) {
            var tot    = s.adv + s.dec || 1;
            var advW   = (s.adv / tot * 100).toFixed(1);
            return '<div class="ib-sec-row">' +
                '<div class="ib-sec-name" onclick="goToSector(\'' + esc(s.name) + '\')">' + esc(s.name) + '</div>' +
                '<div class="ib-sec-bar">' +
                    '<div class="ib-sec-adv" style="width:' + advW + '%"></div>' +
                    '<div class="ib-sec-dec"></div>' +
                '</div>' +
                '<div class="ib-sec-cnt"><span style="color:#3fb950;">' + s.adv + '</span> / <span style="color:#f85149;">' + s.dec + '</span></div>' +
            '</div>';
        }).join('');

        el.innerHTML =
            '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">' +
                '<div class="analytics-title">Industry Breadth</div>' +
                '<div class="analytics-sub">' + total + ' industries</div>' +
            '</div>' +
            '<div class="ib-top">' +
                '<div class="ib-stat"><div class="ib-stat-label">Advancing</div><div class="ib-stat-val up">' + adv + '</div><div class="ib-stat-pct up">' + advPct + '%</div></div>' +
                '<div class="ib-stat c"><div class="ib-stat-label">Unchanged</div><div class="ib-stat-val flat">' + unch + '</div></div>' +
                '<div class="ib-stat r"><div class="ib-stat-label">Declining</div><div class="ib-stat-val down">' + dec + '</div><div class="ib-stat-pct down">' + decPct + '%</div></div>' +
            '</div>' +
            '<div class="ib-master">' +
                '<div class="ib-master-adv" style="width:' + advPct + '%"></div>' +
                (unch ? '<div class="ib-master-unch" style="width:' + (unch/total*100).toFixed(1) + '%"></div>' : '') +
                '<div class="ib-master-dec"></div>' +
            '</div>' +
            secRows +
            '<div style="display:flex;justify-content:space-between;margin-top:10px;padding-top:10px;border-top:1px solid #21262d;">' +
                '<div class="rs-foot-stat"><div class="rs-foot-label">Net</div><div class="rs-foot-val ' + netDir + '">' + (net >= 0 ? '+' : '') + net + '</div></div>' +
                '<div class="rs-foot-stat"><div class="rs-foot-label">A/D Ratio</div><div class="rs-foot-val ' + netDir + '">' + ratio + '</div></div>' +
                '<div class="rs-foot-stat"><div class="rs-foot-label">Best</div><div class="rs-foot-val" style="font-size:11px;color:#3fb950;">' + esc(sectorList[0] ? sectorList[0].name.split(' ')[0] : '—') + '</div></div>' +
                '<div class="rs-foot-stat"><div class="rs-foot-label">Worst</div><div class="rs-foot-val" style="font-size:11px;color:#f85149;">' + esc(sectorList[sectorList.length-1] ? sectorList[sectorList.length-1].name.split(' ')[0] : '—') + '</div></div>' +
            '</div>';
    }

    // ── RS Distribution ───────────────────────────────────────────────────
    function renderRSDist() {
        var el = document.getElementById('market-rs-dist');
        if (!el || !snapshot || !snapshot.by_industry) return;

        var COLORS  = ['#f85149','#f85149','#e3852b','#e3852b','#484f58','#484f58','#8bc34a','#8bc34a','#3fb950','#3fb950'];
        var XLABELS = ['0','10','20','30','40','50','60','70','80','90'];
        var W = 400, H = 160, padL = 36, padB = 22, padT = 8, padR = 4;
        var chartW = W - padL - padR;
        var chartH = H - padB - padT;
        var barW   = chartW / 10;

        // ── Pre-compute both datasets ──────────────────────────────────────
        function computeDataset(field) {
            var buckets = [0,0,0,0,0,0,0,0,0,0];
            var all = [], total = 0;
            Object.values(snapshot.by_industry).forEach(function(rows) {
                rows.forEach(function(r) {
                    var v = r[field];
                    if (v == null) return;
                    var p = Math.min(99, Math.max(0, v));
                    buckets[Math.min(9, Math.floor(p / 10))]++;
                    all.push(p);
                    total++;
                });
            });
            var median = 0, avg = 0;
            if (all.length) {
                all.sort(function(a,b){return a-b;});
                median = Math.round(all[Math.floor(all.length/2)]);
                avg    = Math.round(all.reduce(function(s,v){return s+v;},0) / all.length);
            }
            var above80 = buckets[8] + buckets[9];
            var below20 = buckets[0] + buckets[1];
            var skew    = above80 > below20 * 1.5 ? 'top' : below20 > above80 * 1.5 ? 'bot' : 'bal';
            return { buckets: buckets, total: total, median: median, avg: avg, above80: above80, below20: below20, skew: skew };
        }

        var datasets = {
            rs:   computeDataset('Percentile'),
            rs3m: computeDataset('weighted_rs_pct'),
        };
        var activeDs = 'rs';

        // ── SVG builder (grid + bars only; x-labels are static) ───────────
        function buildSvgInner(ds) {
            var maxBucket = Math.max.apply(null, ds.buckets) || 1;
            var yMax  = Math.ceil(maxBucket / 100) * 100 || 100;
            var yTicks = [yMax, yMax*0.75, yMax*0.5, yMax*0.25, 0];

            var gridHtml = '';
            yTicks.forEach(function(v) {
                var y = padT + chartH - (v / yMax) * chartH;
                gridHtml += '<line x1="' + padL + '" y1="' + y + '" x2="' + (W-padR) + '" y2="' + y + '" stroke="#21262d" stroke-width="1"/>';
                gridHtml += '<text x="' + (padL-4) + '" y="' + (y+3.5) + '" fill="#484f58" font-size="9" text-anchor="end" font-family="inherit">' + v + '</text>';
            });

            var barsHtml = '';
            for (var i = 0; i < 10; i++) {
                var bh = (ds.buckets[i] / yMax) * chartH;
                var bx = padL + i * barW;
                var by = padT + chartH - bh;
                barsHtml += '<rect x="' + (bx+2) + '" y="' + by + '" width="' + (barW-4) + '" height="' + bh + '" fill="' + COLORS[i] + '" rx="2"/>';
            }

            var xlHtml = '';
            for (var j = 0; j < 10; j++) {
                var lx = padL + j * barW + barW / 2;
                xlHtml += '<text x="' + lx + '" y="' + (H-4) + '" fill="#484f58" font-size="9" text-anchor="middle" font-family="inherit">' + XLABELS[j] + '</text>';
            }
            return gridHtml + barsHtml + xlHtml;
        }

        // ── Footer HTML builder ────────────────────────────────────────────
        function buildFooter(ds) {
            var skewLabel = ds.skew === 'top' ? 'Top Heavy' : ds.skew === 'bot' ? 'Bot Heavy' : 'Balanced';
            return '<div class="rs-foot-stat"><div class="rs-foot-label">Median</div><div class="rs-foot-val">' + ds.median + '</div></div>' +
                   '<div class="rs-foot-stat"><div class="rs-foot-label">Avg</div><div class="rs-foot-val">' + ds.avg + '</div></div>' +
                   '<div class="rs-foot-stat"><div class="rs-foot-label">Above 80</div><div class="rs-foot-val up">' + ds.above80 + '</div></div>' +
                   '<div class="rs-foot-stat"><div class="rs-foot-label">Below 20</div><div class="rs-foot-val down">' + ds.below20 + '</div></div>' +
                   '<div class="rs-foot-stat"><div class="rs-foot-label">Skew</div><div class="rs-skew-pill ' + ds.skew + '">' + skewLabel + '</div></div>';
        }

        // ── Initial render ─────────────────────────────────────────────────
        var legend =
            '<div style="display:flex;gap:12px;margin-top:6px;">' +
                '<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:#6e7681;"><div style="width:10px;height:10px;border-radius:2px;background:#3fb950;flex-shrink:0;"></div>≥ 60</div>' +
                '<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:#6e7681;"><div style="width:10px;height:10px;border-radius:2px;background:#484f58;flex-shrink:0;"></div>40–60</div>' +
                '<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:#6e7681;"><div style="width:10px;height:10px;border-radius:2px;background:#f85149;flex-shrink:0;"></div>≤ 40</div>' +
            '</div>';

        var toggleStyle = 'display:inline-flex;border:1px solid #21262d;border-radius:4px;overflow:hidden;margin-left:8px;';
        var btnBase     = 'padding:1px 7px;font-size:10px;font-weight:600;font-family:inherit;cursor:pointer;border:none;letter-spacing:0.03em;transition:background 0.1s,color 0.1s;';
        var btnActive   = btnBase + 'background:#1f6feb;color:#fff;';
        var btnInactive = btnBase + 'background:transparent;color:#6e7681;';

        el.innerHTML =
            '<div style="display:flex;justify-content:space-between;align-items:center;">' +
                '<div style="display:flex;align-items:center;">' +
                    '<div class="analytics-title" id="rs-dist-title">RS Distribution</div>' +
                    '<div style="' + toggleStyle + '">' +
                        '<button id="rs-dist-btn-rs"   style="' + btnActive   + '" onclick="rsDistSwitch(\'rs\')">RS</button>' +
                        '<button id="rs-dist-btn-rs3m" style="' + btnInactive + '" onclick="rsDistSwitch(\'rs3m\')">3M RS</button>' +
                    '</div>' +
                '</div>' +
                '<div class="analytics-sub" id="rs-dist-count">' + datasets.rs.total.toLocaleString() + ' stocks</div>' +
            '</div>' +
            '<svg id="rs-dist-svg" viewBox="0 0 ' + W + ' ' + H + '" width="100%" style="display:block;margin-top:8px;" xmlns="http://www.w3.org/2000/svg">' +
                buildSvgInner(datasets.rs) +
            '</svg>' +
            legend +
            '<div class="rs-footer" id="rs-dist-footer">' + buildFooter(datasets.rs) + '</div>';

        // ── Toggle handler ─────────────────────────────────────────────────
        window.rsDistSwitch = function(which) {
            if (which === activeDs) return;
            activeDs = which;
            var ds = datasets[which];
            var svg = document.getElementById('rs-dist-svg');
            if (svg) svg.innerHTML = buildSvgInner(ds);
            var footer = document.getElementById('rs-dist-footer');
            if (footer) footer.innerHTML = buildFooter(ds);
            var title = document.getElementById('rs-dist-title');
            if (title) title.textContent = which === 'rs3m' ? '3M RS Distribution' : 'RS Distribution';
            var count = document.getElementById('rs-dist-count');
            if (count) count.textContent = ds.total.toLocaleString() + ' stocks';
            var btnRs   = document.getElementById('rs-dist-btn-rs');
            var btnRs3m = document.getElementById('rs-dist-btn-rs3m');
            if (btnRs)   btnRs.style.cssText   = (which === 'rs'   ? btnActive : btnInactive);
            if (btnRs3m) btnRs3m.style.cssText = (which === 'rs3m' ? btnActive : btnInactive);
        };
    }

