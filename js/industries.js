    // ── Industry list click delegation ──────────────────────────────────
    document.getElementById('industry-list').addEventListener('click', function(e) {
        var row = e.target.closest('.industry-row');
        if (!row) return;
        var industry = row.getAttribute('data-industry');
        if (industry) openIndustry(industry);
    });

    // ── Sector drill-down ────────────────────────────────────────────────
    var currentSector = '';
    var sectorSort = { col: null, dir: -1 };

    window.openSector = function(sectorName) {
        currentSector = sectorName;
        var secCl = sectorClass(sectorName);

        document.getElementById('sector-name').innerHTML =
            '<span class="' + secCl + '">' + esc(sectorName) + '</span>';

        // Reset column sort whenever a fresh sector is opened
        sectorSort = { col: null, dir: -1 };
        document.querySelectorAll('#sector-list-header .ind-col-hdr').forEach(function(el){ el.classList.remove('sorted','asc','desc'); });

        renderSectorIndustries();
        showView('sector');
    };

    function renderSectorIndustries() {
        // Get all industries in this sector from industriesData
        var industries = (industriesData && industriesData.industries)
            ? industriesData.industries.filter(function(i){ return i.sector === currentSector; })
            : [];

        if (sectorSort.col) {
            industries.sort(function(a,b){
                var sumA = snapshot && snapshot.industry_summary && snapshot.industry_summary[a.industry];
                var sumB = snapshot && snapshot.industry_summary && snapshot.industry_summary[b.industry];
                var va = getIndSortVal(a, sumA, sectorSort.col);
                var vb = getIndSortVal(b, sumB, sectorSort.col);
                if (va == null && vb == null) return 0;
                if (va == null) return 1;
                if (vb == null) return -1;
                return (va - vb) * sectorSort.dir * -1;
            });
        } else {
            industries.sort(function(a,b){ return (a.rank||999) - (b.rank||999); });
        }

        document.getElementById('sector-meta').textContent =
            industries.length + ' industries';

        var html = '';
        industries.forEach(function(ind) {
            var summary = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry];
            var stockCount = (snapshot && snapshot.by_industry && snapshot.by_industry[ind.industry])
                ? snapshot.by_industry[ind.industry].length : 0;
            html += '<div class="industry-row" data-industry="' + esc(ind.industry) + '" onclick="openIndustry(\'' + esc(ind.industry) + '\')">';
            html += '<span class="industry-rank">' + (ind.rank || '—') + '</span>';
            html += '<div class="industry-name"><span class="industry-name-text">' + esc(ind.industry) + '</span>' + indRankDeltaHtml(ind.industry, ind.rank) + '</div>';
            html += '<span class="industry-count">' + stockCount + '</span>';
            html += '<div class="industry-perf-cols">';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_daily : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1w    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_3m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_6m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1y    : null) + '</span>';
            html += '</div>';
            html += '</div>';
        });
        document.getElementById('sector-industry-list').innerHTML = html || '<div class="loading-msg">No industries.</div>';
    }

    window.setSectorSort = function(col) {
        if (sectorSort.col === col) {
            sectorSort.dir *= -1;
        } else {
            sectorSort.col = col;
            sectorSort.dir = -1; // default desc (best performers first)
        }
        document.querySelectorAll('#sector-list-header .ind-col-hdr').forEach(function(el) {
            el.classList.remove('sorted','asc','desc');
            if (el.getAttribute('data-col') === col) {
                el.classList.add('sorted', sectorSort.dir === 1 ? 'asc' : 'desc');
            }
        });
        renderSectorIndustries();
    };

    window.backFromSector = function() {
        showView('industries');
    };

        // ── Timeframe / sort ──────────────────────────────────────────────────
    window.setIndSort = function(col) {
        if (indSort.col === col) {
            indSort.dir *= -1;
        } else {
            indSort.col = col;
            indSort.dir = -1; // default desc (best performers first)
        }
        // Update header indicators
        document.querySelectorAll('#industry-list-header .ind-col-hdr').forEach(function(el) {
            el.classList.remove('sorted','asc','desc');
            if (el.getAttribute('data-col') === col) {
                el.classList.add('sorted', indSort.dir === 1 ? 'asc' : 'desc');
            }
        });
        renderIndustries();
    };

    window.setSort = function(s) {
        activeSort = s;
        indSort = { col: null, dir: -1 }; // reset column sort when switching preset sort
        document.querySelectorAll('.sort-btn').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-sort') === s); });
        document.querySelectorAll('#industry-list-header .ind-col-hdr').forEach(function(el){ el.classList.remove('sorted','asc','desc'); });
        renderIndustries();
    };

    // ── Industries render ─────────────────────────────────────────────────
    function getIndSortVal(ind, summary, col) {
        if (!col) return null;
        if (col === 'avg_daily') return summary ? summary.avg_daily : null;
        if (col === 'avg_1w')    return summary ? summary.avg_1w    : null;
        if (col === 'avg_1m')    return summary ? summary.avg_1m    : null;
        if (col === 'avg_3m')    return summary ? summary.avg_3m    : null;
        if (col === 'avg_6m')    { var _s6 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return _s6 ? _s6.avg_6m  : null; }
        if (col === 'avg_1y')    { var _s1y = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return _s1y ? _s1y.avg_1y : null; }
        if (col === 'avg_ytd')   return summary ? summary.avg_ytd   : null;
        if (col === 'rs_1m')     { var s1 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s1 ? s1.rs_1m  : null; }
        if (col === 'rs')        { var s2 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s2 ? s2.rs_12m : null; }
        if (col === 'rs_3m')     { var s3 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s3 ? s3.rs_3m  : null; }
        if (col === 'rs_6m')     { var s4 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s4 ? s4.rs_6m  : null; }
        return null;
    }

    function rsColClass(v) { return v == null ? 'neutral' : v >= 50 ? 'positive' : 'negative'; }

    function indRankDeltaHtml(industryName, currentRank) {
        var prev = indPrevRanks[industryName];
        if (prev == null || currentRank == null) return '';
        var delta = prev - currentRank; // positive = moved up (rank number decreased)
        if (delta === 0) return '';
        var cls = delta > 0 ? 'up' : 'down';
        var sign = delta > 0 ? '+' : '';
        return '<span class="ind-rank-delta ' + cls + '">' + sign + delta + '</span>';
    }

    function perfCol(val) {
        if (val == null) return '<span class="industry-perf-num neutral">—</span>';
        var cl = val > 0 ? 'up' : val < 0 ? 'down' : 'neutral';
        return '<span class="industry-perf-num ' + cl + '">' + (val >= 0 ? '+' : '') + val.toFixed(2) + '%</span>';
    }

    function sparkSvg(series) {
        if (!series || series.length < 2) return '<span class="industry-spark-empty">—</span>';
        var w = 100, h = 24, pad = 2;
        var min = Math.min.apply(null, series), max = Math.max.apply(null, series);
        var range = (max - min) || 1;
        var stepX = (w - pad * 2) / (series.length - 1);
        var pts = series.map(function(v, i) {
            var x = pad + i * stepX;
            var y = pad + (1 - (v - min) / range) * (h - pad * 2);
            return [x.toFixed(1), y.toFixed(1)];
        });
        var cl = series[series.length - 1] >= series[0] ? 'up' : 'down';
        var line = 'M' + pts.map(function(p){ return p[0] + ',' + p[1]; }).join(' L');
        var area = line + ' L' + pts[pts.length - 1][0] + ',' + (h - pad) + ' L' + pts[0][0] + ',' + (h - pad) + ' Z';
        var last = pts[pts.length - 1];
        return '<svg class="industry-spark ' + cl + '" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">' +
            '<path class="industry-spark-area" d="' + area + '"></path>' +
            '<path class="industry-spark-path" d="' + line + '"></path>' +
            '<circle class="industry-spark-dot" cx="' + last[0] + '" cy="' + last[1] + '" r="2"></circle>' +
            '</svg>';
    }


    // ── Industry heatmap ──────────────────────────────────────────────────
    var indView    = 'list';
    var heatmapTf  = 'avg_daily';

    function heatmapColor(val) {
        if (val == null) return null;
        // Scale: 0% = neutral, ±5% = full saturation (clamp beyond)
        var maxVal = 5.0;
        var t = Math.min(1, Math.abs(val) / maxVal);
        if (val > 0) {
            // green: from #1a3a2a (near 0) to #1a7a3a (full)
            var g = Math.round(60 + t * 100);
            var r = Math.round(10 + t * 5);
            var b = Math.round(20 + t * 10);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            // red: from #3a1a1a (near 0) to #8a1a1a (full)
            var r2 = Math.round(60 + t * 100);
            var g2 = Math.round(10 + t * 5);
            var b2 = Math.round(10 + t * 5);
            return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
        }
    }

    function heatmapValLabel(tf) {
        var map = { avg_daily:'Day', avg_1w:'1W', avg_1m:'1M', avg_3m:'3M', avg_6m:'6M', avg_1y:'1Y', avg_ytd:'YTD' };
        return map[tf] || tf;
    }

    function renderHeatmap() {
        var container = document.getElementById('industry-heatmap');
        if (!industriesData || !industriesData.industries) {
            container.innerHTML = '<div class="loading-msg">No industry data.</div>';
            return;
        }
        var industries = industriesData.industries.slice();
        var q = searchQuery.toLowerCase();
        if (q) industries = industries.filter(function(i){ return i.industry.toLowerCase().includes(q) || i.sector.toLowerCase().includes(q); });

        // Sort by selected timeframe descending
        industries.sort(function(a, b) {
            var sumA = snapshot && snapshot.industry_summary && snapshot.industry_summary[a.industry];
            var sumB = snapshot && snapshot.industry_summary && snapshot.industry_summary[b.industry];
            var va = sumA ? sumA[heatmapTf] : null;
            var vb = sumB ? sumB[heatmapTf] : null;
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            return vb - va;
        });

        document.getElementById('heatmap-result-count').textContent = industries.length + ' industries';

        var html = '<div class="heatmap-grid">';
        industries.forEach(function(ind) {
            var summary = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry];
            var val = summary ? summary[heatmapTf] : null;
            var bg  = heatmapColor(val);
            var valStr = val != null ? (val >= 0 ? '+' : '') + val.toFixed(2) + '%' : '—';
            var isNull = bg == null;
            html += '<div class="heatmap-card' + (isNull ? ' heatmap-card-null' : '') + '"' +
                    ' style="background:' + (bg || '#161b22') + ';"' +
                    ' data-industry="' + esc(ind.industry) + '"' +
                    ' onclick="openIndustry(\'' + esc(ind.industry) + '\')">' +
                    '<div class="heatmap-card-name">' + esc(ind.industry) + '</div>' +
                    '<div class="heatmap-card-val">' + valStr + '</div>' +
                    '</div>';
        });
        html += '</div>';
        container.innerHTML = html;
    }

    window.setIndView = function(view) {
        indView = view;
        var isList = view === 'list';

        document.getElementById('industry-list-header').style.display = isList ? 'flex' : 'none';
        document.getElementById('heatmap-toolbar').style.display      = isList ? 'none' : 'flex';
        document.getElementById('industry-list').style.display        = isList ? 'block' : 'none';
        document.getElementById('industry-heatmap').style.display     = isList ? 'none' : 'block';

        // Sync toggle buttons
        ['ind-view-list-btn','ind-view-list-btn2'].forEach(function(id) {
            var el = document.getElementById(id); if (el) el.classList.toggle('active', isList);
        });
        ['ind-view-heat-btn','ind-view-heat-btn2'].forEach(function(id) {
            var el = document.getElementById(id); if (el) el.classList.toggle('active', !isList);
        });

        if (!isList) renderHeatmap();
        else renderIndustries();
    };

    window.setHeatmapTf = function(tf) {
        heatmapTf = tf;
        document.querySelectorAll('.heatmap-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-htf') === tf);
        });
        renderHeatmap();
    };

    function renderIndustries() {
        var list = document.getElementById('industry-list');
        if (!industriesData || !industriesData.industries) {
            list.innerHTML = '<div class="loading-msg">No industry data.</div>';
            return;
        }
        var industries = industriesData.industries.slice();
        var q = searchQuery.toLowerCase();
        if (q) industries = industries.filter(function(i){ return i.industry.toLowerCase().includes(q) || i.sector.toLowerCase().includes(q); });

        // Sort by column if active, else by preset
        if (indSort.col) {
            industries.sort(function(a, b) {
                var sumA = snapshot && snapshot.industry_summary && snapshot.industry_summary[a.industry];
                var sumB = snapshot && snapshot.industry_summary && snapshot.industry_summary[b.industry];
                var va = getIndSortVal(a, sumA, indSort.col);
                var vb = getIndSortVal(b, sumB, indSort.col);
                if (va == null && vb == null) return 0;
                if (va == null) return 1;
                if (vb == null) return -1;
                return (va - vb) * indSort.dir * -1;
            });
        } else {
            if (activeSort === 'rank')   industries.sort(function(a,b){ return (a.rank||999) - (b.rank||999); });
            else if (activeSort === 'name')   industries.sort(function(a,b){ return a.industry.localeCompare(b.industry); });
            else if (activeSort === 'sector') industries.sort(function(a,b){ return a.sector.localeCompare(b.sector) || (a.rank||999) - (b.rank||999); });
        }

        var rcEl = document.getElementById('result-count'); if (rcEl) rcEl.textContent = industries.length + ' industries';

        var html = '';
        industries.forEach(function(ind) {
            var stockCount = (snapshot && snapshot.by_industry && snapshot.by_industry[ind.industry])
                ? snapshot.by_industry[ind.industry].length : (ind.tickers ? ind.tickers.length : 0);
            var summary = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry];
            var _indSum = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry];
            var rs1m  = _indSum ? _indSum.rs_1m  : null;
            var rs3m  = _indSum ? _indSum.rs_3m  : null;
            var rs6m  = _indSum ? _indSum.rs_6m  : null;
            var rs12m = _indSum ? _indSum.rs_12m : null;
            var rsCurr = rs12m;

            html += '<div class="industry-row" data-industry="' + esc(ind.industry) + '">';
            html += '<span class="industry-rank">' + (ind.rank || '—') + '</span>';
            html += '<div class="industry-name"><span class="industry-name-text">' + esc(ind.industry) + '</span>' + indRankDeltaHtml(ind.industry, ind.rank) + '</div>';
            html += '<span class="industry-spark-col">' + sparkSvg(summary ? summary.spark_3m : null) + '</span>';
            html += '<span class="industry-flex-fill"></span>';
            html += '<span class="industry-sector ' + sectorClass(ind.sector) + '">' + esc(ind.sector) + '</span>';
            html += '<span class="industry-count">' + stockCount + '</span>';
            // Perf columns
            html += '<div class="industry-perf-cols">';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_daily : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1w    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_3m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_6m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1y    : null) + '</span>';
            html += '</div>';
            // RS columns
            html += '<div class="industry-rs-cols">';
            html += '<span class="industry-rs-col ' + rsColClass(rsCurr) + '">' + (rsCurr != null ? rsCurr : '—') + '</span>';
            html += '<span class="industry-rs-col ' + rsColClass(rs1m) + '">' + (rs1m != null ? rs1m : '—') + '</span>';
            html += '<span class="industry-rs-col ' + rsColClass(rs3m) + '">' + (rs3m != null ? rs3m : '—') + '</span>';
            html += '<span class="industry-rs-col ' + rsColClass(rs6m) + '">' + (rs6m != null ? rs6m : '—') + '</span>';
            html += '</div>';

            html += '</div>';
        });
        list.innerHTML = html || '<div class="loading-msg">No results.</div>';
    }

    // ── Open industry → stocks ────────────────────────────────────────────
    window.openIndustry = function(industryName) {
        _lastIndustryName      = industryName;
        _lastIndustryScrollTop = 0;
        var _mainArea = document.getElementById('main-area');
        if (_mainArea) _industriesListScrollTop = _mainArea.scrollTop;
        var ind  = industriesData && industriesData.industries.find(function(i){ return i.industry === industryName; });
        var rows = snapshot && snapshot.by_industry && snapshot.by_industry[industryName];

        var totalIndustries = (industriesData && industriesData.industries) ? industriesData.industries.length : 0;
        var rank = ind ? (ind.rank || '—') : '—';
        document.getElementById('si-name').innerHTML =
            esc(industryName) +
            '<span style="font-size:0.78em;font-weight:400;color:#6e7681;margin-left:10px;">' +
            '(' + rank + '/' + totalIndustries + ')</span>';
        var sector = ind ? ind.sector : '';
        var secCl  = sectorClass(sector);
        var metaEl = document.getElementById('si-meta');
        if (sector) {
            metaEl.innerHTML = '<span id="si-sector-link" class="' + secCl + '" style="cursor:pointer;font-weight:600;">' + esc(sector) + '</span>';
            document.getElementById('si-sector-link').onclick = function() { openSector(sector); };
        } else {
            metaEl.innerHTML = '';
        }
        document.getElementById('si-rs').textContent = ind ? 'RS ' + (ind.percentile || '—') : '';

        // Reset multichart state
        multichartActive = false;
        mcWidgets = {};
        document.getElementById('stocks-table-view').style.display      = 'flex';
        document.getElementById('stocks-multichart-view').style.display = 'none';
        document.getElementById('multichart-toggle-btn').style.background  = '';
        document.getElementById('multichart-toggle-btn').style.borderColor = '';
        document.getElementById('multichart-toggle-btn').style.color       = '';
        document.getElementById('multichart-grid').innerHTML = '';

        currentStockSort = { by: 'weighted_rs_pct', dir: 1, count: 1 };
        selectedIndustryStocks.clear();
        indUpdateExportBtn();

        if (rows && rows.length > 0) {
            rows = rows.slice().sort(function(a, b) {
                var av = a.weighted_rs_pct != null ? a.weighted_rs_pct : -Infinity;
                var bv = b.weighted_rs_pct != null ? b.weighted_rs_pct : -Infinity;
                return bv - av;
            });
        }
        mcTickers = rows ? rows.map(function(r){ return r.ticker; }) : [];

        if (!rows || rows.length === 0) {
            document.getElementById('stocks-thead').innerHTML = '';
            document.getElementById('stocks-tbody').innerHTML =
                '<tr><td colspan="12" style="padding:30px;text-align:center;color:#484f58;">' +
                'No stock data for this industry yet — run the build script to populate.</td></tr>';
        } else {
            renderStocksTable(rows, industryName, 'stocks-thead', 'stocks-tbody');
        }
        showView('industry-stocks');
        if (rows && rows.length) {
            indStartPricePolling(rows.map(function(r) { return r.ticker; }));
        }
    };
