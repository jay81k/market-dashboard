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
            var rs1m  = summary ? summary.rs_1m  : null;
            var rs3m  = summary ? summary.rs_3m  : null;
            var rs6m  = summary ? summary.rs_6m  : null;
            var rs12m = summary ? summary.rs_12m : null;
            var rsCurr = rs12m;
            var pctUptrend = summary ? summary.pct_confirmed_uptrend : null;
            var pctAbove50 = summary ? summary.pct_above_50sma       : null;
            var pctAbove21 = summary ? summary.pct_above_21ema       : null;

            html += '<div class="industry-row" data-industry="' + esc(ind.industry) + '" onclick="openIndustry(\'' + esc(ind.industry) + '\')">';
            html += '<span class="industry-rank">' + (ind.rank || '—') + '</span>';
            html += '<div class="industry-name"><span class="industry-name-text">' + esc(ind.industry) + '</span>' + indRankDeltaHtml(ind.industry, ind.rank) + '</div>';
            html += topRsChipsHtml(ind.industry);
            html += '<span class="industry-spark-col">' + sparkSvg(summary ? summary.spark_3m : null) + '</span>';
            html += '<span class="industry-count">' + stockCount + '</span>';
            html += '<div class="industry-perf-cols">';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_daily : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1w    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_3m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_6m    : null) + '</span>';
            html += '<span class="industry-perf-col">' + perfCol(summary ? summary.avg_1y    : null) + '</span>';
            html += '</div>';
            html += '<div class="industry-rs-cols">';
            html += '<span class="industry-rs-col ' + rsColClass(rsCurr) + '">' + (rsCurr != null ? rsCurr : '—') + '</span>';
            html += '<span class="industry-rs-col ' + rsColClass(rs1m)  + '">' + (rs1m  != null ? rs1m  : '—') + '</span>';
            html += '<span class="industry-rs-col ' + rsColClass(rs3m)  + '">' + (rs3m  != null ? rs3m  : '—') + '</span>';
            html += '<span class="industry-rs-col ' + rsColClass(rs6m)  + '">' + (rs6m  != null ? rs6m  : '—') + '</span>';
            html += '</div>';
            html += '<div class="industry-breadth-cols">';
            html += '<span class="industry-breadth-col ' + rsColClass(pctUptrend) + '">' + (pctUptrend != null ? pctUptrend + '%' : '—') + '</span>';
            html += '<span class="industry-breadth-col ' + rsColClass(pctAbove50) + '">' + (pctAbove50 != null ? pctAbove50 + '%' : '—') + '</span>';
            html += '<span class="industry-breadth-col ' + rsColClass(pctAbove21) + '">' + (pctAbove21 != null ? pctAbove21 + '%' : '—') + '</span>';
            html += '</div>';
            html += '</div>';
        });
        document.getElementById('sector-industry-list').innerHTML = html || '<div class="loading-msg">No industries.</div>';
        if (typeof tickerHoverBind === 'function') tickerHoverBind(document.getElementById('sector-industry-list'), '.industry-chip', null);
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
        if (col === 'pct_confirmed_uptrend') { var s5 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s5 ? s5.pct_confirmed_uptrend : null; }
        if (col === 'pct_above_50sma')       { var s6 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s6 ? s6.pct_above_50sma : null; }
        if (col === 'pct_above_21ema')       { var s7 = snapshot && snapshot.industry_summary && snapshot.industry_summary[ind.industry]; return s7 ? s7.pct_above_21ema : null; }
        return null;
    }

    function rsColClass(v) { return v == null ? 'neutral' : v >= 50 ? 'positive' : 'negative'; }

    // ── Top-5-by-3M-RS ticker chips (industry list + sector drill-down) ────
    // "3M RS" here is the per-ticker weighted_rs_pct field — the same one
    // labeled "3M RS" on the stocks table and the industry-detail RS badge.
    // That's distinct from the industry-level rs_3m summary field used by
    // the list's own "3M RS" sort column.
    function topRsChipsHtml(industryName) {
        var rows = snapshot && snapshot.by_industry && snapshot.by_industry[industryName];
        if (!rows || !rows.length) return '<div class="industry-chips"></div>';
        var top = rows.slice().sort(function(a, b) {
            var av = a.weighted_rs_pct != null ? a.weighted_rs_pct : -Infinity;
            var bv = b.weighted_rs_pct != null ? b.weighted_rs_pct : -Infinity;
            return bv - av;
        }).slice(0, 5);
        var html = '<div class="industry-chips">';
        top.forEach(function(r) {
            html += '<span class="industry-chip"' +
                ' onclick="event.stopPropagation();openChartModal(\'' + esc(r.ticker) + '\')"' +
                ' oncontextmenu="event.preventDefault();event.stopPropagation();indChipContext(event,\'' + esc(r.ticker) + '\')">' +
                esc(r.ticker) + '</span>';
        });
        html += '</div>';
        return html;
    }

    // Same fakeBtn contract wlOpenPicker expects everywhere else it's called
    // from a right-click (stocks.js tbody, market-popup.js popup).
    window.indChipContext = function(e, ticker) {
        var fakeBtn = {
            getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
            getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
            _wlNoSwitch: true
        };
        wlOpenPicker(fakeBtn, e, false);
    };

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

    // Same channel math as heatmapColor, but reduced to a luminance figure so the
    // rank watermark can pick black vs white tint (and how strong) per actual card color,
    // instead of one flat opacity guessed to work "on average".
    function rankTintStyle(val) {
        if (val == null) return 'rgba(255,255,255,0.16)';
        var maxVal = 5.0;
        var t = Math.min(1, Math.abs(val) / maxVal);
        var r, g, b;
        if (val > 0) {
            g = 60 + t * 100; r = 10 + t * 5; b = 20 + t * 10;
        } else {
            r = 60 + t * 100; g = 10 + t * 5; b = 10 + t * 5;
        }
        var luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if (luminance > 90) {
            var opBlack = Math.min(0.30, 0.18 + (luminance - 90) / 120 * 0.12);
            return 'rgba(0,0,0,' + opBlack.toFixed(2) + ')';
        }
        var opWhite = Math.max(0.14, 0.22 - luminance / 90 * 0.06);
        return 'rgba(255,255,255,' + opWhite.toFixed(2) + ')';
    }

    function renderHeatmap() {
        var container = document.getElementById('industry-heatmap');
        if (!industriesData || !industriesData.industries) {
            hideHeatLeaders();
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
            var rankStr = (ind.rank != null) ? String(ind.rank) : '';
            var rankFontSize = rankStr.length >= 3 ? 28 : (rankStr.length === 2 ? 36 : 44);
            var rankWatermark = rankStr ? (
                '<span style="position:absolute;right:8px;bottom:4px;font-size:' + rankFontSize + 'px;font-weight:500;color:' + rankTintStyle(val) + ';line-height:1;">' + rankStr + '</span>'
            ) : '';
            html += '<div class="heatmap-card' + (isNull ? ' heatmap-card-null' : '') + '"' +
                    ' style="background:' + (bg || 'var(--bg-subtle)') + ';position:relative;overflow:hidden;min-height:84px;"' +
                    ' data-industry="' + esc(ind.industry) + '"' +
                    ' onclick="openIndustry(\'' + esc(ind.industry) + '\')">' +
                    rankWatermark +
                    '<div style="position:relative;z-index:1;display:flex;flex-direction:column;justify-content:space-between;height:100%;">' +
                    '<div class="heatmap-card-name">' + esc(ind.industry) + '</div>' +
                    '<div class="heatmap-card-val">' + valStr + '</div>' +
                    '</div>' +
                    '</div>';
        });
        html += '</div>';
        container.innerHTML = html;
        bindHeatLeaders(container);
        heatLeadersReattach(container);
    }

    // ── Heatmap hover: leaders quick view ─────────────────────────────────
    // Hovering a heatmap card opens a side panel with that industry's top
    // tickers by 3M RS — the per-ticker weighted_rs_pct field, same sort as
    // topRsChipsHtml() above. The panel is one fixed-position element on
    // <body> so it survives renderHeatmap()'s innerHTML rebuilds and isn't
    // clipped by the grid's scroll area.
    var HEAT_LEADERS_N        = 5;
    var HEAT_LEADERS_SHOW_DAY = false;   // true adds a Day % column (widens the panel)
    var HEAT_LEADERS_OPEN_MS  = 160;     // hover-intent delay before opening
    var HEAT_LEADERS_CLOSE_MS = 140;     // grace period to travel card -> panel
    var heatLeadersEl       = null;
    var heatLeadersCard     = null;
    var heatLeadersIndustry = null;
    var heatLeadersOpenT    = null;
    var heatLeadersCloseT   = null;

    function heatLeadersPanelHtml(industryName) {
        var rows = snapshot && snapshot.by_industry && snapshot.by_industry[industryName];
        if (!rows || !rows.length) return '';
        var top = rows.slice().sort(function(a, b) {
            var av = a.weighted_rs_pct != null ? a.weighted_rs_pct : -Infinity;
            var bv = b.weighted_rs_pct != null ? b.weighted_rs_pct : -Infinity;
            return bv - av;
        }).slice(0, HEAT_LEADERS_N);

        var html = '<div class="hl-head">' +
                   '<div class="hl-title">' + esc(industryName) + '</div>' +
                   '<div class="hl-sub">Top ' + top.length + ' by 3M RS · ' + rows.length + ' stocks</div>' +
                   '</div>';
        top.forEach(function(r, i) {
            var rs    = r.weighted_rs_pct != null ? Number(r.weighted_rs_pct) : NaN;
            var hasRs = !isNaN(rs);
            // Same tiers as the RS badges (rs-high / rs-mid / rs-low)
            var tier  = !hasRs ? '' : rs >= 75 ? ' rs-high' : rs >= 40 ? ' rs-mid' : ' rs-low';
            var barW  = hasRs ? Math.max(0, Math.min(100, rs)).toFixed(0) : 0;
            html += '<div class="hl-row' + tier + '" data-ticker="' + esc(r.ticker) + '">' +
                    '<span class="hl-idx">' + (i + 1) + '</span>' +
                    '<span class="hl-tkr">' + esc(r.ticker) + '</span>' +
                    '<span class="hl-bar"><span style="width:' + barW + '%"></span></span>' +
                    '<span class="hl-rs">' + (hasRs ? Math.round(rs) : '—') + '</span>';
            if (HEAT_LEADERS_SHOW_DAY) {
                var d = r.daily != null ? Number(r.daily) : NaN;
                html += isNaN(d)
                    ? '<span class="hl-day">—</span>'
                    : '<span class="hl-day ' + (d > 0 ? 'up' : d < 0 ? 'down' : '') + '">' + (d >= 0 ? '+' : '') + d.toFixed(2) + '%</span>';
            }
            html += '</div>';
        });
        html += '<div class="hl-foot">Click: chart · Right-click: watchlist</div>';
        return html;
    }

    function ensureHeatLeadersEl() {
        if (heatLeadersEl) return heatLeadersEl;
        var el = document.createElement('div');
        el.id = 'heat-leaders';
        el.className = 'heat-leaders';
        document.body.appendChild(el);

        el.addEventListener('mouseenter', function() { clearTimeout(heatLeadersCloseT); });
        el.addEventListener('mouseleave', scheduleHideHeatLeaders);
        el.addEventListener('click', function(e) {
            var row = e.target.closest('.hl-row');
            if (!row) return;
            var t = row.getAttribute('data-ticker');
            hideHeatLeaders();
            if (t) openChartModal(t);
        });
        // Same right-click -> watchlist picker as the Leaders chips in the list view
        el.addEventListener('contextmenu', function(e) {
            var row = e.target.closest('.hl-row');
            if (!row) return;
            e.preventDefault();
            e.stopPropagation();
            var t = row.getAttribute('data-ticker');
            if (t) window.indChipContext(e, t);
        });

        // A fixed panel would drift away from its card if anything scrolled/resized
        var hideIfOpen = function() { if (heatLeadersEl && heatLeadersEl.style.display === 'block') hideHeatLeaders(); };
        document.addEventListener('scroll', hideIfOpen, true);
        window.addEventListener('resize', hideIfOpen);

        heatLeadersEl = el;
        return el;
    }

    function positionHeatLeaders(card) {
        var el = heatLeadersEl;
        var r  = card.getBoundingClientRect();
        var pw = el.offsetWidth, ph = el.offsetHeight;
        // Slight overlap with the card edge so the cursor never crosses dead space
        var left = r.right - 2;
        if (left + pw > window.innerWidth - 8) left = r.left - pw + 2;
        left = Math.max(8, left);
        var top = Math.max(8, Math.min(r.top, window.innerHeight - ph - 8));
        el.style.left = left + 'px';
        el.style.top  = top  + 'px';
    }

    function showHeatLeaders(card) {
        if (!document.body.contains(card)) return;
        var name = card.getAttribute('data-industry');
        var html = heatLeadersPanelHtml(name);
        if (!html) return;
        var el = ensureHeatLeadersEl();
        if (heatLeadersCard && heatLeadersCard !== card) heatLeadersCard.classList.remove('heat-active');
        heatLeadersCard     = card;
        heatLeadersIndustry = name;
        card.classList.add('heat-active');
        el.className = 'heat-leaders' + (HEAT_LEADERS_SHOW_DAY ? ' has-day' : '');
        el.innerHTML = html;
        el.style.visibility = 'hidden';   // measure before showing
        el.style.display    = 'block';
        positionHeatLeaders(card);
        el.style.visibility = '';
    }

    function hideHeatLeaders() {
        clearTimeout(heatLeadersOpenT);
        clearTimeout(heatLeadersCloseT);
        if (heatLeadersEl) heatLeadersEl.style.display = 'none';
        if (heatLeadersCard) heatLeadersCard.classList.remove('heat-active');
        heatLeadersCard = null;
        heatLeadersIndustry = null;
    }

    function scheduleHideHeatLeaders() {
        clearTimeout(heatLeadersCloseT);
        heatLeadersCloseT = setTimeout(hideHeatLeaders, HEAT_LEADERS_CLOSE_MS);
    }

    // renderHeatmap() also runs on the live-day refresh (state.js) and rebuilds
    // every card. If the panel is open, re-anchor it to the rebuilt card
    // instead of leaving it attached to a detached node.
    function heatLeadersReattach(container) {
        if (!heatLeadersIndustry || !heatLeadersEl || heatLeadersEl.style.display !== 'block') return;
        var cards = container.querySelectorAll('.heatmap-card');
        for (var i = 0; i < cards.length; i++) {
            if (cards[i].getAttribute('data-industry') === heatLeadersIndustry) { showHeatLeaders(cards[i]); return; }
        }
        hideHeatLeaders();
    }

    // Delegated once on the persistent container (its children are rebuilt on every render)
    function bindHeatLeaders(container) {
        if (container._heatLeadersBound) return;
        container._heatLeadersBound = true;

        container.addEventListener('mouseover', function(e) {
            var card = e.target.closest('.heatmap-card');
            if (!card) { scheduleHideHeatLeaders(); return; }          // grid gap / padding
            if (e.relatedTarget && card.contains(e.relatedTarget)) return;  // moving inside the same card
            clearTimeout(heatLeadersCloseT);
            if (card === heatLeadersCard) return;
            clearTimeout(heatLeadersOpenT);
            heatLeadersOpenT = setTimeout(function() { showHeatLeaders(card); }, HEAT_LEADERS_OPEN_MS);
        });
        container.addEventListener('mouseout', function(e) {
            var card = e.target.closest('.heatmap-card');
            if (!card) return;
            if (e.relatedTarget && card.contains(e.relatedTarget)) return;
            clearTimeout(heatLeadersOpenT);
            scheduleHideHeatLeaders();
        });
        // Capture phase: close before the card's own onclick navigates away
        container.addEventListener('click', hideHeatLeaders, true);
    }

    window.setIndView = function(view) {
        indView = view;
        hideHeatLeaders();
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

    window.refreshHeatmapData = function() {
        var btn = document.getElementById('heatmap-refresh-btn');
        if (btn) { btn.classList.add('spinning'); btn.disabled = true; }
        fetchLiveIndustryDay(function() {
            if (btn) { btn.classList.remove('spinning'); btn.disabled = false; }
        });
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
            html += topRsChipsHtml(ind.industry);
            html += '<span class="industry-sector ' + sectorClass(ind.sector) + '">' + esc(ind.sector) + '</span>';
            html += '<span class="industry-spark-col">' + sparkSvg(summary ? summary.spark_3m : null) + '</span>';
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
            // Breadth columns
            var pctUptrend = _indSum ? _indSum.pct_confirmed_uptrend : null;
            var pctAbove50 = _indSum ? _indSum.pct_above_50sma       : null;
            var pctAbove21 = _indSum ? _indSum.pct_above_21ema       : null;
            html += '<div class="industry-breadth-cols">';
            html += '<span class="industry-breadth-col ' + rsColClass(pctUptrend) + '">' + (pctUptrend != null ? pctUptrend + '%' : '—') + '</span>';
            html += '<span class="industry-breadth-col ' + rsColClass(pctAbove50) + '">' + (pctAbove50 != null ? pctAbove50 + '%' : '—') + '</span>';
            html += '<span class="industry-breadth-col ' + rsColClass(pctAbove21) + '">' + (pctAbove21 != null ? pctAbove21 + '%' : '—') + '</span>';
            html += '</div>';

            html += '</div>';
        });
        list.innerHTML = html || '<div class="loading-msg">No results.</div>';
        if (typeof tickerHoverBind === 'function') tickerHoverBind(list, '.industry-chip', null);
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
        // Rank 1 is best, rank totalIndustries is worst — convert to a
        // 0-100 scale (100 = best) so the same 75/40 thresholds used for
        // every RS percentile in this app apply here too.
        var rankColor = 'var(--border-muted)';
        if (ind && ind.rank && totalIndustries > 1) {
            var rankPct = ((totalIndustries - ind.rank) / (totalIndustries - 1)) * 100;
            rankColor = rankPct >= 75 ? 'var(--success)' : rankPct >= 40 ? 'var(--warning-alt)' : 'var(--danger)';
        }
        document.getElementById('si-name').innerHTML =
            esc(industryName) +
            '<span style="font-size:0.78em;font-weight:400;color:' + rankColor + ';margin-left:10px;">' +
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
        var rsPct = ind ? ind.percentile : null;
        var rsEl  = document.getElementById('si-rs');
        rsEl.textContent = 'RS ' + (rsPct != null ? Math.round(rsPct) : '—');
        rsEl.className   = 'stocks-rs-badge' + (rsPct != null ? (rsPct >= 75 ? ' rs-high' : rsPct >= 40 ? ' rs-mid' : ' rs-low') : '');

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
                '<tr><td colspan="12" style="padding:30px;text-align:center;color:var(--border-muted);">' +
                'No stock data for this industry yet — run the build script to populate.</td></tr>';
        } else {
            renderStocksTable(rows, industryName, 'stocks-thead', 'stocks-tbody');
        }
        showView('industry-stocks');
        if (rows && rows.length) {
            indStartPricePolling(rows.map(function(r) { return r.ticker; }));
        }
    };
