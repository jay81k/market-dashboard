    // ── Industry-stocks live prices ───────────────────────────────────────
    var indLivePrices  = {};   // { ticker: { price, prevClose } }
    var indPriceTimer  = null;

    function indFetchPrices(tickers) {
        if (!tickers || !tickers.length) return;
        var batches = [];
        for (var i = 0; i < tickers.length; i += 50) batches.push(tickers.slice(i, i + 50));
        batches.forEach(function(batch) {
            var url = WL_PROXY + '?action=quotes_batch&tickers=' + batch.map(encodeURIComponent).join(',');
            fetch(url).then(function(r) { return r.ok ? r.json() : null; }).then(function(data) {
                if (!data || !data.quotes) return;
                data.quotes.forEach(function(q) {
                    if (q && q.ticker && q.price) {
                        indLivePrices[q.ticker] = { price: q.price, prevClose: q.prevClose || null, dayHigh: q.dayHigh || null, dayLow: q.dayLow || null };
                    }
                });
                indUpdatePriceRows();
            }).catch(function() {});
        });
    }

    function indUpdatePriceRows() {
        document.querySelectorAll('#stocks-tbody .stock-row').forEach(function(tr) {
            var ticker = tr.getAttribute('data-symbol');
            var live   = indLivePrices[ticker];
            if (!live || !live.price) return;
            var price     = live.price;
            var prevClose = live.prevClose;
            var chgAbs    = (prevClose && prevClose > 0) ? price - prevClose : null;
            var chgPct    = (prevClose && prevClose > 0) ? ((price - prevClose) / prevClose) * 100 : null;
            var cl        = chgPct == null ? '' : chgPct > 0 ? 'up' : chgPct < 0 ? 'down' : '';
            var tds       = tr.querySelectorAll('td');
            if (tds[2]) tds[2].textContent = '$' + price.toFixed(2);
            if (tds[5]) { tds[5].textContent = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '\u2014'; tds[5].className = cl; }
            if (tds[6]) { tds[6].textContent = chgPct != null ? (chgPct >= 0 ? '+' : '') + chgPct.toFixed(2) + '%' : '\u2014'; tds[6].className = cl; }
            if (tds[17] && live.dayHigh != null && live.dayLow != null && live.dayHigh > live.dayLow) {
                var liveCr = ((price - live.dayLow) / (live.dayHigh - live.dayLow)) * 100;
                var liveCrColor = liveCr >= 60 ? '#3fb950' : liveCr >= 30 ? '#e3852b' : '#f85149';
                tds[17].innerHTML = '<span style="color:' + liveCrColor + ';font-weight:600;">' + Math.round(liveCr) + '%</span>';
                tr.setAttribute('data-cr', liveCr);
            }
            tr.setAttribute('data-price', price);
            if (chgAbs != null) tr.setAttribute('data-chg', chgAbs.toFixed(4));
            if (chgPct != null) tr.setAttribute('data-daily', chgPct);
            // ── Live Dist/MA ──────────────────────────────────────────────
            var distCell = tr.querySelector('.dist-ma-cell');
            if (distCell) {
                var snapRow     = tickerMap && tickerMap[ticker];
                var snapPrice   = snapRow ? (snapRow._snapPrice != null ? snapRow._snapPrice : snapRow.price) : null;
                var snapDistAll = snapRow ? snapRow.dist_ma  : null;
                var dmaKey      = activeMAType + activeMALength;
                if (snapPrice && snapDistAll) {
                    // Recompute live dist for every MA key and persist to
                    // data-dist-all so applyDistMA (called on sort / MA change)
                    // never overwrites live values with stale snapshot values.
                    var liveDistAll = {};
                    Object.keys(snapDistAll).forEach(function(k) {
                        var sd = snapDistAll[k];
                        if (sd != null) {
                            var maVal = snapPrice / (1 + sd / 100);
                            liveDistAll[k] = (price - maVal) / maVal * 100;
                        }
                    });
                    distCell.setAttribute('data-dist-all', JSON.stringify(liveDistAll));
                    var liveDist = liveDistAll[dmaKey];
                    if (liveDist != null) {
                        var ldCl = liveDist > 0 ? 'up' : liveDist < 0 ? 'down' : '';
                        distCell.innerHTML = '<span class="' + ldCl + '">' + fmt(liveDist, 2, '%') + '</span>';
                        tr.setAttribute('data-dist_ma', liveDist);
                    }
                }
            }
            // ── Live multichart candle update ─────────────────────────────
            _updateMcLiveCandle(ticker, price, live.dayHigh, live.dayLow, mcWidgets);
        });
    }

    function indStartPricePolling(tickers) {
        if (indPriceTimer) clearInterval(indPriceTimer);
        indLivePrices = {};
        indFetchPrices(tickers);
        if (!wlIsMarketOpen()) return;
        indPriceTimer = setInterval(function() {
            if (currentView !== 'industry-stocks') { indStopPricePolling(); return; }
            if (!wlIsMarketOpen()) { indStopPricePolling(); return; }
            indFetchPrices(tickers);
        }, 60 * 1000);
    }

    function indStopPricePolling() {
        if (indPriceTimer) { clearInterval(indPriceTimer); indPriceTimer = null; }
    }

    // ── Render stocks table ───────────────────────────────────────────────
    function buildTableHeader(theadId, industryName) {
        var dmaTooltip = '% distance from ' + activeMAType + activeMALength;
        var cols = [
            { key:'symbol',     label:'Ticker',   tip:'Ticker symbol' },
            { key:'price',      label:'Price',     tip:'Last closing price' },
            { key:'rs',         label:'RS',        tip:'RS Percentile (1–99)' },
            { key:'weighted_rs_pct', label:'3M RS',  tip:'Weighted 3M RS Percentile' },
            { key:'chg',        label:'Chg',       tip:'Daily change ($)' },
            { key:'daily',      label:'Chg%',      tip:'Daily return %' },
            { key:'1w',         label:'1W',        tip:'5-day return' },
            { key:'1m',         label:'1M',        tip:'21-day return' },
            { key:'3m',         label:'3M',        tip:'63-day return' },
            { key:'ytd',        label:'1Y',        tip:'1-year return' },
            { key:'vs_spy',     label:'vs 1M',     tip:'1M return vs SPX' },
            { key:'vs_spy_3m',  label:'vs 3M',     tip:'3M return vs SPX' },
            { key:'dist_ma',    label:'Dist/MA',   tip:dmaTooltip, extra:'<span class="dist-ma-btn" onclick="event.stopPropagation();toggleDistMA(this)">⋯</span>' },
            { key:'avg_vol',    label:'Avg Vol',   tip:'50-day average volume' },
            { key:'pct_52wk',   label:'52Wk%',     tip:'% from 52-week high' },
            { key:'adr_pct',    label:'ADR%',      tip:'Avg Daily Range %' },
            { key:'cr',         label:'CR',        tip:'Closing range (100=high, 0=low)' },
        ];
        var fundCols = [
            { key:'fwd_pe',            label:'PE',        tip:'Forward P/E ratio' },
            { key:'ps_ratio',          label:'P/S',       tip:'Price / Sales ratio' },
            { key:'peg_ratio',         label:'PEG',       tip:'Price / Earnings / Growth ratio' },
            { key:'eps_this_y_pct',    label:'EPS TY',    tip:'EPS Growth This Year %' },
            { key:'eps_next_y_pct',    label:'EPS NY',    tip:'EPS Growth Next Year %' },
            { key:'eps_next_5y_pct',   label:'EPS 5Y',    tip:'EPS Growth Next 5 Years %' },
            { key:'eps_qoq_pct',       label:'EPS Q/Q',   tip:'EPS Growth Qtr over Qtr %' },
            { key:'sales_qoq_pct',     label:'Sales Q/Q', tip:'Sales Growth Qtr over Qtr %' },
            { key:'profit_margin_pct', label:'Margin',    tip:'Profit Margin %' },
        ];
        var html = '<tr>';
        html += '<th style="width:28px;padding-left:8px;cursor:pointer;" title="Select / deselect all" onclick="indToggleSelectAll()">' +
            '<input type="checkbox" id="ind-select-all-chk" onclick="event.stopPropagation();indToggleSelectAll()" style="cursor:pointer;accent-color:#388bfd;color-scheme:dark;opacity:0.35;">' +
            '</th>';
        cols.forEach(function(c) {
            var sortCl = (currentStockSort.by === c.key) ? (' sorted ' + (currentStockSort.dir === 1 ? 'sort-desc' : 'sort-asc')) : '';
            var label = c.key === 'symbol' ? '' : c.label;
            html += '<th class="sortable' + sortCl + '" data-sort-by="' + c.key + '" data-tooltip="' + esc(c.tip) + '">' + label + (c.extra||'') + '</th>';
        });
        fundCols.forEach(function(c) {
            var sortCl = (currentStockSort.by === c.key) ? (' sorted ' + (currentStockSort.dir === 1 ? 'sort-desc' : 'sort-asc')) : '';
            html += '<th class="sortable' + sortCl + '" data-sort-by="' + c.key + '" data-tooltip="' + esc(c.tip) + '">' + c.label + '</th>';
        });
        html += '</tr>';
        document.getElementById(theadId).innerHTML = html;
    }

    function renderStocksTable(rows, industryName, theadId, tbodyId) {
        buildTableHeader(theadId, industryName);
        var maKey = activeMAType + activeMALength;

        var html = '';
        rows.forEach(function(row, i) {
            var distAll = row.dist_ma || {};
            var distVal = distAll[maKey];
            var rsVal   = row.Percentile != null ? Math.round(row.Percentile) : '—';
            var crVal   = row.cr  != null ? row.cr.toFixed(0)  + '%' : '—';
            var adrVal  = row.adr_pct != null ? row.adr_pct.toFixed(2) + '%' : '—';

            html += '<tr class="stock-row"' +
                    ' data-symbol="' + esc(row.ticker) + '"' +
                    ' data-index="' + i + '"' +
                    ' data-rs="'      + (row.Percentile != null ? row.Percentile : '')  + '"' +
                    ' data-weighted_rs_pct="' + (row.weighted_rs_pct != null ? row.weighted_rs_pct : '') + '"' +
                    ' data-price="'   + (row.price      != null ? row.price      : '')  + '"' +
                    ' data-daily="'   + (row.daily      != null ? row.daily      : '')  + '"' +
                    ' data-chg="'     + (row.price != null && row.daily != null ? ((row.price/(1+row.daily/100))*(row.daily/100)).toFixed(4) : '') + '"' +
                    ' data-1w="'      + (row['1w']      != null ? row['1w']      : '')  + '"' +
                    ' data-1m="'      + (row['1m']      != null ? row['1m']      : '')  + '"' +
                    ' data-3m="'      + (row['3m']      != null ? row['3m']      : '')  + '"' +
                    ' data-ytd="'     + (row['1y']       != null ? row['1y']       : '')  + '"' +
                    ' data-vs_spy="'  + (row.vs_spy     != null ? row.vs_spy     : '')  + '"' +
                    ' data-vs_spy_3m="'+(row.vs_spy_3m  != null ? row.vs_spy_3m  : '') + '"' +
                    ' data-dist_ma="' + (distVal        != null ? distVal        : '')  + '"' +
                    ' data-avg_vol="' + (row.AvgVol50   != null ? row.AvgVol50   : '')  + '"' +
                    ' data-pct_52wk="'+ (row.PctFrom52WkHigh != null ? row.PctFrom52WkHigh : '') + '"' +
                    ' data-adr_pct="' + (row.adr_pct    != null ? row.adr_pct    : '')  + '"' +
                    ' data-cr="'      + (row.cr         != null ? row.cr         : '')  + '"' +
                    ' data-fwd_pe="'        + (row.fwd_pe        != null ? row.fwd_pe        : '') + '"' +
                    ' data-ps_ratio="'      + (row.ps_ratio      != null ? row.ps_ratio      : '') + '"' +
                    ' data-peg_ratio="'     + (row.peg_ratio     != null ? row.peg_ratio     : '') + '"' +
                    ' data-eps_this_y_pct="'     + (row.eps_this_y_pct    != null ? row.eps_this_y_pct    : '') + '"' +
                    ' data-eps_next_y_pct="'    + (row.eps_next_y_pct    != null ? row.eps_next_y_pct    : '') + '"' +
                    ' data-eps_next_5y_pct="'   + (row.eps_next_5y_pct   != null ? row.eps_next_5y_pct   : '') + '"' +
                    ' data-eps_qoq_pct="'       + (row.eps_qoq_pct       != null ? row.eps_qoq_pct       : '') + '"' +
                    ' data-sales_qoq_pct="'     + (row.sales_qoq_pct     != null ? row.sales_qoq_pct     : '') + '"' +
                    ' data-profit_margin_pct="' + (row.profit_margin_pct != null ? row.profit_margin_pct : '') + '"' +
                    ' data-sector="'  + esc(row.sector   || '') + '"' +
                    ' data-industry="'+ esc(row.industry || '') + '"' +
                    '>';
            var _isAdded = selectedIndustryStocks.has(row.ticker);
            html += '<td onclick="event.stopPropagation()" style="padding-left:8px;width:28px;">' +
                '<button class="scan-add-btn' + (_isAdded ? ' added' : '') + '" data-ticker="' + esc(row.ticker) + '" onclick="event.stopPropagation();indToggleAdd(this)" title="Select for export">' +
                (_isAdded ? SVG_IND_CHECK : SVG_IND_PLUS) + '</button></td>';
            html += '<td style="white-space:nowrap;"><button class="wl-add-btn" data-ticker="' + esc(row.ticker) + '" onclick="event.stopPropagation();wlQuickToggle(this)" title="Add to watchlist">☆</button><button class="wl-pick-btn" data-ticker="' + esc(row.ticker) + '" onclick="event.stopPropagation();wlOpenPicker(this,event)" title="Choose watchlist">▾</button><span class="ticker-badge">' + esc(row.ticker) + '</span></td>';
            // Price
            var priceVal = row.price != null ? '$' + row.price.toFixed(2) : '—';
            html += '<td style="color:#c8d0dc;font-weight:500;">' + priceVal + '</td>';
            html += '<td style="color:#c8d0dc;font-weight:600;">' + rsVal + '</td>';
            var wrsVal = row.weighted_rs_pct != null ? Math.round(row.weighted_rs_pct) : '—';
            html += '<td style="color:#c8d0dc;font-weight:600;">' + wrsVal + '</td>';
            // Chg ($) and Chg%
            var chgAbs = (row.price != null && row.daily != null) ? (row.price / (1 + row.daily / 100)) * (row.daily / 100) : null;
            var chgStr = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '—';
            html += '<td class="' + cc(row.daily) + '">' + chgStr + '</td>';
            html += '<td class="' + cc(row.daily) + '">' + (row.daily != null ? fmt(row.daily,2,'%') : '—') + '</td>';
            html += '<td class="' + cc(row['1w'])      + '">' + (row['1w']      != null ? fmt(row['1w'],2,'%')      : '—') + '</td>';
            html += '<td class="' + cc(row['1m'])      + '">' + (row['1m']      != null ? fmt(row['1m'],2,'%')      : '—') + '</td>';
            html += '<td class="' + cc(row['3m'])      + '">' + (row['3m']      != null ? fmt(row['3m'],2,'%')      : '—') + '</td>';
            html += '<td class="' + cc(row['1y'])      + '">' + (row['1y']      != null ? fmt(row['1y'],2,'%')      : '—') + '</td>';
            html += '<td class="' + cc(row.vs_spy)     + '">' + (row.vs_spy     != null ? fmt(row.vs_spy,2,'%')     : '—') + '</td>';
            html += '<td class="' + cc(row.vs_spy_3m)  + '">' + (row.vs_spy_3m  != null ? fmt(row.vs_spy_3m,2,'%')  : '—') + '</td>';
            html += '<td class="dist-ma-cell" data-dist-all="' + esc(JSON.stringify(distAll)) + '"></td>';
            // Avg Vol (from CSV AvgVol50)
            var volVal = '—';
            if (row.AvgVol50 != null) {
                var v = row.AvgVol50;
                volVal = v >= 1e6 ? (v/1e6).toFixed(1) + 'M' : v >= 1e3 ? (v/1e3).toFixed(0) + 'K' : v.toFixed(0);
            }
            html += '<td style="color:#8b949e;">' + volVal + '</td>';
            var pct52Val = '—';
            var pct52Color = '#484f58';
            if (row.PctFrom52WkHigh != null) {
                pct52Val = (row.PctFrom52WkHigh > 0 ? '+' : '') + row.PctFrom52WkHigh.toFixed(1) + '%';
                pct52Color = row.PctFrom52WkHigh >= -5 ? '#3fb950' : row.PctFrom52WkHigh >= -15 ? '#e3852b' : '#f85149';
            }
            html += '<td><span style="color:' + pct52Color + ';font-weight:600;">' + pct52Val + '</span></td>';
            var adrColor = '#484f58';
            if (row.adr_pct != null) {
                if (row.adr_pct < 4)      adrColor = '#3fb950';
                else if (row.adr_pct < 8) adrColor = '#e3852b';
                else                       adrColor = '#f85149';
            }
            html += '<td><span style="color:' + adrColor + ';font-weight:600;">' + adrVal + '</span></td>';
            var crColor = '#484f58';
            if (row.cr != null) {
                if (row.cr >= 60) crColor = '#3fb950';
                else if (row.cr >= 30) crColor = '#e3852b';
                else crColor = '#f85149';
            }
            html += '<td><span style="color:' + crColor + ';font-weight:600;">' + crVal + '</span></td>';
            // Valuation columns (lower = better — color logic inverted)
            var fwdPeColor = '#484f58';
            if (row.fwd_pe != null && row.fwd_pe > 0) {
                if (row.fwd_pe < 15)       fwdPeColor = '#3fb950';
                else if (row.fwd_pe <= 25) fwdPeColor = '#8b949e';
                else                       fwdPeColor = '#f85149';
            }
            html += '<td><span style="color:' + fwdPeColor + ';font-weight:600;">' + (row.fwd_pe != null && row.fwd_pe > 0 ? row.fwd_pe.toFixed(2) : '—') + '</span></td>';
            var psColor = '#484f58';
            if (row.ps_ratio != null && row.ps_ratio >= 0) {
                if (row.ps_ratio < 2)      psColor = '#3fb950';
                else if (row.ps_ratio <= 5) psColor = '#8b949e';
                else                        psColor = '#f85149';
            }
            html += '<td><span style="color:' + psColor + ';font-weight:600;">' + (row.ps_ratio != null ? row.ps_ratio.toFixed(2) : '—') + '</span></td>';
            var pegColor = '#484f58';
            if (row.peg_ratio != null && row.peg_ratio > 0) {
                if (row.peg_ratio < 1)      pegColor = '#3fb950';
                else if (row.peg_ratio <= 2) pegColor = '#8b949e';
                else                         pegColor = '#f85149';
            }
            html += '<td><span style="color:' + pegColor + ';font-weight:600;">' + (row.peg_ratio != null && row.peg_ratio > 0 ? row.peg_ratio.toFixed(2) : '—') + '</span></td>';
            // Fundamental columns
            html += '<td class="' + cc(row.eps_this_y_pct)    + '">' + (row.eps_this_y_pct    != null ? fmt(row.eps_this_y_pct,2,'%')    : '—') + '</td>';
            html += '<td class="' + cc(row.eps_next_y_pct)    + '">' + (row.eps_next_y_pct    != null ? fmt(row.eps_next_y_pct,2,'%')    : '—') + '</td>';
            html += '<td class="' + cc(row.eps_next_5y_pct)   + '">' + (row.eps_next_5y_pct   != null ? fmt(row.eps_next_5y_pct,2,'%')   : '—') + '</td>';
            html += '<td class="' + cc(row.eps_qoq_pct)       + '">' + (row.eps_qoq_pct       != null ? fmt(row.eps_qoq_pct,2,'%')       : '—') + '</td>';
            html += '<td class="' + cc(row.sales_qoq_pct)     + '">' + (row.sales_qoq_pct     != null ? fmt(row.sales_qoq_pct,2,'%')     : '—') + '</td>';
            html += '<td class="' + cc(row.profit_margin_pct) + '">' + (row.profit_margin_pct != null ? fmt(row.profit_margin_pct,2,'%') : '—') + '</td>';
            html += '</tr>';
        });
        var tbody = document.getElementById(tbodyId);
        tbody.innerHTML = html;
        if (typeof tickerHoverBind === 'function') tickerHoverBind(tbody, '.ticker-badge', null);
        applyDistMA(tbody);

        // Attach click/dblclick via delegation
        tbody.onclick = function(e) {
            var row = e.target.closest('.stock-row');
            if (!row) return;
            if (e.target.closest('.wl-add-btn') || e.target.closest('.wl-pick-btn') || e.target.closest('.scan-add-btn')) return;
            tbody.querySelectorAll('.stock-row.active').forEach(function(r){ r.classList.remove('active'); });
            row.classList.add('active');
            allStockRows = Array.from(tbody.querySelectorAll('.stock-row'));
            currentStockIndex = allStockRows.indexOf(row);
        };
        tbody.ondblclick = function(e) {
            var row = e.target.closest('.stock-row');
            if (!row) return;
            openChartModal(row.getAttribute('data-symbol'));
        };
        tbody.oncontextmenu = function(e) {
            var row = e.target.closest('.stock-row');
            if (!row) return;
            e.preventDefault();
            var ticker = row.getAttribute('data-symbol');
            var fakeBtn = {
                getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
                getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                _wlNoSwitch: true
            };
            wlOpenPicker(fakeBtn, e, false);
        };

        allStockRows = Array.from(tbody.querySelectorAll('.stock-row'));
        currentStockIndex = -1;
        wlRefreshStars();
        if (typeof alStampBadges === 'function') alStampBadges();
    }

    // ── Industry-drill stocks table sorting ─────────────────────────────────
    document.getElementById('main-area').addEventListener('click', function(e) {
        var th = e.target.closest('th.sortable');
        if (!th || e.target.closest('.dist-ma-btn')) return;
        var thead = th.closest('thead');
        var table = th.closest('table');

        // Only handle industry-drill stocks table here; scans has its own handler
        if (!table || table.id === 'scans-table') return;

        var sortBy = th.getAttribute('data-sort-by');
        var tbody  = document.getElementById('stocks-tbody');
        if (!tbody) return;
        if (currentStockSort.by === sortBy) {
            currentStockSort.count++;
            if (currentStockSort.count >= 3) {
                currentStockSort = { by: null, dir: 1, count: 0 };
                var rows = Array.from(tbody.querySelectorAll('.stock-row'));
                rows.sort(function(a,b){ return parseInt(a.getAttribute('data-index')||0) - parseInt(b.getAttribute('data-index')||0); });
                tbody.innerHTML = ''; rows.forEach(function(r){ tbody.appendChild(r); });
                applyDistMA(tbody);
                thead.querySelectorAll('th').forEach(function(t){ t.classList.remove('sort-asc','sort-desc','sorted'); });
                allStockRows = rows;
                mcTickers = rows.map(function(r){ return r.getAttribute('data-symbol'); });
                if (multichartActive) renderMulticharts();
                return;
            }
        } else { currentStockSort.count = 1; }
        currentStockSort.by = sortBy;
        if (currentStockSort.count === 2) currentStockSort.dir *= -1; else currentStockSort.dir = 1;
        thead.querySelectorAll('th').forEach(function(t){ t.classList.remove('sort-asc','sort-desc','sorted'); });
        th.classList.add(currentStockSort.dir === 1 ? 'sort-desc' : 'sort-asc', 'sorted');
        var rows = Array.from(tbody.querySelectorAll('.stock-row'));
        rows.sort(function(a,b) {
            if (sortBy === 'symbol') return a.getAttribute('data-symbol').localeCompare(b.getAttribute('data-symbol')) * currentStockSort.dir;
            var av = parseFloat(a.getAttribute('data-' + sortBy));
            var bv = parseFloat(b.getAttribute('data-' + sortBy));            if (isNaN(av)) return 1; if (isNaN(bv)) return -1;
            return (av - bv) * currentStockSort.dir;
        });
        tbody.innerHTML = ''; rows.forEach(function(r){ tbody.appendChild(r); });
        applyDistMA(tbody);
        allStockRows = rows;
        mcTickers = rows.map(function(r){ return r.getAttribute('data-symbol'); });
        if (multichartActive) renderMulticharts();
    });

        // ── Dist/MA ───────────────────────────────────────────────────────────
    function maKey() { return activeMAType + activeMALength; }

    function applyDistMA(container) {
        var key = maKey();
        var cells = (container || document).querySelectorAll('.dist-ma-cell');
        cells.forEach(function(cell) {
            var raw = cell.getAttribute('data-dist-all');
            if (!raw || raw.length < 3) { cell.innerHTML = '<span style="color:#30363d">—</span>'; return; }
            var obj = null;
            try { obj = JSON.parse(raw); } catch(e) {
                try { obj = JSON.parse(raw.replace(/&quot;/g, '"')); } catch(e2) {}
            }
            var val = obj ? obj[key] : null;
            if (val == null) { cell.innerHTML = '<span style="color:#30363d">—</span>'; return; }
            var cl = val > 0 ? 'up' : val < 0 ? 'down' : '';
            cell.innerHTML = '<span class="' + cl + '">' + fmt(val,2,'%') + '</span>';
            var row = cell.closest('tr');
            if (row) row.setAttribute('data-dist_ma', val);
        });
    }

    window.toggleDistMA = function(btn) {
        var dd = document.getElementById('dist-ma-dropdown');
        if (dd.classList.contains('open')) { dd.classList.remove('open'); return; }
        var rect = btn.getBoundingClientRect();
        dd.style.top  = (rect.bottom + 4) + 'px';
        dd.style.left = Math.max(4, rect.right - 190) + 'px';
        dd.classList.add('open');
    };
    document.addEventListener('click', function(e) {
        var dd = document.getElementById('dist-ma-dropdown');
        if (!dd.classList.contains('open')) return;
        if (!dd.contains(e.target) && !e.target.classList.contains('dist-ma-btn')) dd.classList.remove('open');
    });
    window.setDistMAType = function(t) {
        activeMAType = t; localStorage.setItem('distMAType', t);
        document.querySelectorAll('.dist-ma-type-btn').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-type') === t); });
        applyDistMA();
        scanUpdatePriceRows();
        indUpdatePriceRows(); // re-apply live prices to industry stocks table
    };
    window.setDistMALength = function(l) {
        activeMALength = l; localStorage.setItem('distMALength', String(l));
        document.querySelectorAll('.dist-ma-len-btn').forEach(function(b){ b.classList.toggle('active', parseInt(b.getAttribute('data-len')) === l); });
        applyDistMA();
        scanUpdatePriceRows();
        indUpdatePriceRows(); // re-apply live prices to industry stocks table
    };

    // ── Chart modal ───────────────────────────────────────────────────────
    function rsBadge(percentile) {
        if (percentile == null) return null;
        var pct = Math.round(percentile);
        var cls = pct >= 75 ? 'rs-high' : pct >= 40 ? 'rs-mid' : 'rs-low';
        return { text: 'RS ' + pct, cls: cls };
    }

    function applyRsBadge(el, percentile, weightedRsPct, el3m) {
        if (!el) return;
        var b = rsBadge(percentile);
        if (b) {
            el.className = 'chart-rs-badge ' + b.cls;
            el.textContent = b.text;
            el.style.display = '';
        } else {
            el.style.display = 'none';
        }
        if (el3m) {
            if (weightedRsPct != null) {
                var wrs = Math.round(weightedRsPct);
                var wCls = wrs >= 75 ? 'rs-high' : wrs >= 40 ? 'rs-mid' : 'rs-low';
                el3m.className = 'chart-rs-badge ' + wCls;
                el3m.textContent = wrs;
                el3m.style.display = '';
            } else {
                el3m.style.display = 'none';
            }
        }
    }

    function industryLinkHtml(industry, closeFn) {
        if (!industry) return '';
        var escaped = esc(industry);
        var style = 'color:#8b949e;cursor:pointer;border-bottom:1px solid transparent;transition:color 0.15s,border-color 0.15s;';
        var closeFnAttr = closeFn ? ' data-close-fn="' + closeFn + '"' : '';
        return '<span class="industry-nav-link" data-industry-name="' + escaped + '"' + closeFnAttr + ' style="' + style + '">' + escaped + '</span>';
    }

    document.addEventListener('click', function(e) {
        var el = e.target.closest('.industry-nav-link');
        if (!el) return;
        var industry = el.getAttribute('data-industry-name');
        var closeFn  = el.getAttribute('data-close-fn');
        if (closeFn && window[closeFn]) window[closeFn]();
        if (industry) openIndustry(industry);
    });
    document.addEventListener('mouseover', function(e) {
        var el = e.target.closest('.industry-nav-link');
        if (!el) return;
        el.style.color = '#c8d0dc';
        el.style.borderBottomColor = '#6e7681';
    });
    document.addEventListener('mouseout', function(e) {
        var el = e.target.closest('.industry-nav-link');
        if (!el) return;
        el.style.color = '#8b949e';
        el.style.borderBottomColor = 'transparent';
    });

    function fundStatsHtml(row) {
        if (!row) return '';
        var stats = [
            { label: 'EPS Q/Q',  val: row.eps_qoq_pct,       pct: true,  tip: 'EPS Growth Quarter over Quarter' },
            { label: 'Sales Q/Q',val: row.sales_qoq_pct,     pct: true,  tip: 'Sales Growth Quarter over Quarter' },
            { label: 'EPS TY',   val: row.eps_this_y_pct,    pct: true,  tip: 'EPS Growth This Year (estimate)' },
            { label: 'EPS NY',   val: row.eps_next_y_pct,    pct: true,  tip: 'EPS Growth Next Year (estimate)' },
            { label: 'EPS 5Y',   val: row.eps_next_5y_pct,   pct: true,  tip: 'EPS Growth Next 5 Years (annual estimate)' },
            { label: 'Margin',   val: row.profit_margin_pct, pct: false, tip: 'Net Profit Margin' },
        ];
        var hasAny = stats.some(function(s){ return s.val != null; });
        if (!hasAny) return '';
        var divider = '<span style="display:inline-block;width:1px;height:12px;background:#21262d;margin:0 4px;vertical-align:middle;flex-shrink:0;"></span>';
        var html = divider;
        stats.forEach(function(s, i) {
            if (s.val == null) return;
            var v = parseFloat(s.val);
            var color = !s.pct ? (v >= 5 ? '#3fb950' : v < 0 ? '#f85149' : '#8b949e') : (v >= 0 ? '#3fb950' : '#f85149');
            var sign  = s.pct && v > 0 ? '+' : '';
            html += '<span style="display:inline-flex;align-items:center;gap:3px;flex-shrink:0;">' +
                '<span title="' + s.tip + '" style="font-size:0.748em;color:#6e7681;text-transform:uppercase;letter-spacing:0.04em;cursor:default;">' + s.label + '</span>' +
                '<span style="font-size:0.858em;font-weight:600;color:' + color + ';font-variant-numeric:tabular-nums;">' + sign + v.toFixed(1) + '%</span>' +
            '</span>';
            if (i < stats.length - 1) html += divider;
        });
        return html;
    }

    // ── Chart modal symbol right-click → watchlist / alert picker ───────────
    (function() {
        function attachSymRightClick(elId) {
            var symEl = document.getElementById(elId);
            if (!symEl) return;
            symEl.style.cursor = 'context-menu';
            symEl.title = 'Right-click for options';
            symEl.addEventListener('contextmenu', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var ticker = symEl.textContent.trim();
                if (!ticker || ticker === '—') return;
                var fakeBtn = {
                    getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
                    getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                    _wlNoSwitch: true
                };
                wlOpenPicker(fakeBtn, e, false);
                // Opened via contextmenu — no click event fires, so the
                // justOpened guard would eat the first outside click. Reset it.
                wlPickerJustOpened = false;
            });
        }
        attachSymRightClick('wl-chart-sym');
        attachSymRightClick('al-chart-sym');
        attachSymRightClick('mc-fullscreen-sym');
    })();

    document.addEventListener('keydown', function(e) {
        // Always handle Escape first
        if (e.key === 'Escape') {
            if (document.getElementById('mc-fullscreen-overlay').classList.contains('open')) {
                closeChartModal(); return;
            }
            if (currentView === 'industry-stocks') { backToIndustries(); return; }
            return;
        }

        // Don't hijack input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

        var modalOpen = document.getElementById('mc-fullscreen-overlay').classList.contains('open');

        // ── Industries view ───────────────────────────────────────────────
        if (currentView === 'industries' || currentView === 'sector') {
            allIndustryRows = Array.from(document.querySelectorAll('.industry-row'));
            if (!allIndustryRows.length) return;
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                currentIndustryIndex = Math.min(currentIndustryIndex + 1, allIndustryRows.length - 1);
                allIndustryRows.forEach(function(r){ r.style.background = ''; });
                allIndustryRows[currentIndustryIndex].style.background = '#1c2128';
                allIndustryRows[currentIndustryIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                currentIndustryIndex = Math.max(currentIndustryIndex - 1, 0);
                allIndustryRows.forEach(function(r){ r.style.background = ''; });
                allIndustryRows[currentIndustryIndex].style.background = '#1c2128';
                allIndustryRows[currentIndustryIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'Enter' && currentIndustryIndex >= 0) {
                e.preventDefault();
                var indRow = allIndustryRows[currentIndustryIndex];
                var ind = indRow.getAttribute('data-industry');
                if (ind) openIndustry(ind);
            }
            return;
        }

        // ── Industry stocks + Scans views ────────────────────────────────
        if (currentView === 'industry-stocks' || currentView === 'scans') {
            if (!allStockRows.length) return;
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                if (e.key === 'ArrowDown') currentStockIndex = Math.min(currentStockIndex + 1, allStockRows.length - 1);
                else                       currentStockIndex = Math.max(currentStockIndex - 1, 0);

                if (currentView === 'scans') {
                    // Virtual scroll: scroll the wrap to bring the row into view, then highlight
                    var rowData = _vsData[currentStockIndex];
                    if (!rowData) return;
                    var wrap = document.querySelector('#scans-table-view .stocks-table-wrap');
                    if (wrap) {
                        var targetScrollTop = currentStockIndex * _vsRowHeight;
                        var wrapH = wrap.clientHeight;
                        var curScroll = wrap.scrollTop;
                        // Scroll only if row is outside visible area
                        if (targetScrollTop < curScroll + 40) {
                            wrap.scrollTop = Math.max(0, targetScrollTop - 40);
                        } else if (targetScrollTop + _vsRowHeight > curScroll + wrapH - 40) {
                            wrap.scrollTop = targetScrollTop + _vsRowHeight - wrapH + 40;
                        }
                        // After scroll re-render, highlight the correct DOM row
                        setTimeout(function() {
                            var tbody = document.getElementById('scans-tbody');
                            tbody.querySelectorAll('.stock-row.active').forEach(function(x){ x.classList.remove('active'); });
                            var topSpacer = document.getElementById('vs-top-spacer');
                            var topRows   = topSpacer ? Math.round(topSpacer.offsetHeight / _vsRowHeight) : 0;
                            var visRows   = Array.from(tbody.querySelectorAll('.stock-row'));
                            var localIdx  = currentStockIndex - topRows;
                            if (visRows[localIdx]) visRows[localIdx].classList.add('active');
                            if (modalOpen) openChartModal(rowData.ticker);
                        }, 0);
                    }
                } else {
                    var r = allStockRows[currentStockIndex];
                    allStockRows.forEach(function(x){ x.classList.remove('active'); });
                    r.classList.add('active');
                    r.scrollIntoView({ block: 'nearest' });
                    if (modalOpen) openChartModal(r.getAttribute('data-symbol'));
                }
            } else if (e.key === 'Enter' && currentStockIndex >= 0) {
                e.preventDefault();
                if (currentView === 'scans') {
                    var rowData = _vsData[currentStockIndex];
                    if (rowData) openChartModal(rowData.ticker);
                } else {
                    var r = allStockRows[currentStockIndex];
                    openChartModal(r.getAttribute('data-symbol'));
                }
            }
            return;
        }

        // ── Watchlists view ───────────────────────────────────────────────
        if (currentView === 'watchlists') {
            var wlRows = Array.from(document.querySelectorAll('.wl-ticker-row'));
            if (!wlRows.length) return;
            // Sync index with visually active row in case it drifted (e.g. after view switch)
            if (currentWlIndex < 0) {
                var _wlActiveIdx = wlRows.findIndex(function(r) { return r.classList.contains('active'); });
                if (_wlActiveIdx >= 0) currentWlIndex = _wlActiveIdx;
            }
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                currentWlIndex = Math.min(currentWlIndex + 1, wlRows.length - 1);
                var r = wlRows[currentWlIndex];
                r.scrollIntoView({ block: 'nearest' });
                wlSelectTicker(r.getAttribute('data-wl-ticker'));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                currentWlIndex = Math.max(currentWlIndex - 1, 0);
                var r = wlRows[currentWlIndex];
                r.scrollIntoView({ block: 'nearest' });
                wlSelectTicker(r.getAttribute('data-wl-ticker'));
            } else if (e.key === 'Enter' && currentWlIndex >= 0) {
                e.preventDefault();
                var r = wlRows[currentWlIndex];
                var ticker = r.getAttribute('data-wl-ticker');
                if (ticker) openChartModal(ticker);
            } else if (e.key === 'Delete' && currentWlIndex >= 0) {
                // Don't delete ticker if the WL chart has an AVWAP/trendline selected or trendline mode is consuming the key
                if (_wlSelectedVwapIdx !== -1 || _wlSelectedTrendlineIdx !== -1 || (_wlTrendlineMode && _wlTrendlines.length)) return;
                e.preventDefault();
                var r = wlRows[currentWlIndex];
                var ticker   = r.getAttribute('data-wl-ticker');
                var listName = r.getAttribute('data-wl-list');
                if (ticker && listName) {
                    var savedIdx = currentWlIndex;
                    wlRemoveTicker(listName, ticker);
                    var newRows = Array.from(document.querySelectorAll('.wl-ticker-row'));
                    if (newRows.length) {
                        currentWlIndex = Math.min(savedIdx, newRows.length - 1);
                        var next = newRows[currentWlIndex];
                        next.scrollIntoView({ block: 'nearest' });
                        wlSelectTicker(next.getAttribute('data-wl-ticker'));
                    } else {
                        currentWlIndex = -1;
                    }
                }
            }
            return;
        }
    });

    // ── Search ────────────────────────────────────────────────────────────
    var _searchClearBtn = document.getElementById('search-clear-btn');
    function _updateSearchClear(val) {
        _searchClearBtn.classList.toggle('visible', val.length > 0);
    }
    function _applySearchQuery(q) {
        if (currentView === 'industries') {
            searchQuery = q;
            if (indView === 'heatmap') renderHeatmap();
            else renderIndustries();
        } else if (currentView === 'industry-stocks') {
            filterStocksTable(q);
        } else if (currentView === 'scans') {
            filterScansTable(q);
        } else if (currentView === 'watchlists') {
            wlSearchQuery(q);
        }
    }
    document.getElementById('search-input').addEventListener('input', function() {
        var q = this.value;
        _updateSearchClear(q);
        clearTimeout(this._searchTimer);
        this._searchTimer = setTimeout(function() {
            _applySearchQuery(q);
        }, 150);
    });
    _searchClearBtn.addEventListener('click', function() {
        var el = document.getElementById('search-input');
        el.value = '';
        _updateSearchClear('');
        _applySearchQuery('');
        el.focus();
    });

    // ── Column tooltips ───────────────────────────────────────────────────
    (function() {
        var tip = document.getElementById('col-tooltip');
        var timer;
        document.addEventListener('mouseover', function(e) {
            var el = e.target.closest('[data-tooltip]');
            if (!el) return;
            clearTimeout(timer);
            var rect = el.getBoundingClientRect();
            tip.textContent = el.getAttribute('data-tooltip');
            tip.style.left = Math.min(rect.left, window.innerWidth - 200) + 'px';
            tip.style.top  = (rect.bottom + 4) + 'px';
            tip.classList.add('visible');
        });
        document.addEventListener('mouseout', function(e) {
            if (!e.target.closest('[data-tooltip]')) return;
            timer = setTimeout(function(){ tip.classList.remove('visible'); }, 80);
        });
    })();


