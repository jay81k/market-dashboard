    // ── Scans ─────────────────────────────────────────────────────────────
    var scansMultichartActive = false;
    var activeScan = null;
    var scansMcTimeframe    = 'D';
    var scansMcCols         = parseInt(localStorage.getItem('mcSharedCols') || '4');
    var scansMcTickers      = [];
    var scansMcWidgets      = {};
    var scansSortState      = { by: 'weighted_rs_pct', dir: -1 };
    var _indRankMap         = {};
    var scanLivePrices      = {};   // { ticker: { price, prevClose } }
    var scanPriceTimer      = null;
    var _scanLiveRefilterScheduled = false;
    var _scanPreserveScroll        = false;
    var _scanLiveRefilterRender    = false;

    function getAllStocks() {
        if (!snapshot || !snapshot.by_industry) return [];
        var all = [];
        Object.values(snapshot.by_industry).forEach(function(rows) {
            rows.forEach(function(r) { all.push(r); });
        });
        return all;
    }

    // ── Scan filters (dynamic rows) ───────────────────────────────────────
    var sfRows  = [];   // array of filter rule objects
    var sfRowId = 0;
    var sfActivePillType = null;
    var sfActivePopoverId = null;
    var sfPillWasPreExisting = false;
    var sfPopoverIsNew = false;

    var SF_TYPES = [
        { value: 'ma',    label: 'Moving Average' },
        { value: 'rs',    label: 'RS' },
        { value: 'perf',  label: 'Performance' },
        { value: 'price', label: 'Price' },
        { value: 'vol',   label: 'Avg Vol' },
        { value: 'adr',   label: 'ADR%' },
        { value: 'cr',    label: 'CR%' },
        { value: 'mcap',  label: 'Market Cap' },
        { value: 'rvol',    label: 'Rel. Volume' },
        { value: 'udv',     label: 'U/D Vol' },
        { value: 'pattern', label: 'Pattern' },
        { value: 'fund',      label: 'Fundamental' },
        { value: 'valuation', label: 'Valuation' },
        { value: 'gap',     label: 'Gap' },
        { value: 'range',   label: 'Range' },
        { value: 'sector',  label: 'Sector' },
        { value: 'indrank', label: 'Industry Rank' },
    ];

    var FUND_METRICS = [
        { value: 'eps_this_y_pct',    label: 'EPS This Year %' },
        { value: 'eps_next_y_pct',    label: 'EPS Next Year %' },
        { value: 'eps_next_5y_pct',   label: 'EPS Next 5Y %' },
        { value: 'eps_qoq_pct',       label: 'EPS Q/Q %' },
        { value: 'sales_qoq_pct',     label: 'Sales Q/Q %' },
        { value: 'profit_margin_pct', label: 'Profit Margin %' },
    ];

    var VAL_METRICS = [
        { value: 'fwd_pe',    label: 'Forward P/E' },
        { value: 'ps_ratio',  label: 'P/S' },
        { value: 'peg_ratio', label: 'PEG' },
    ];

    function sfParseVol(str) {
        str = String(str || '').trim().toUpperCase().replace(/,/g,'');
        if (str.endsWith('M')) return parseFloat(str) * 1e6;
        if (str.endsWith('K')) return parseFloat(str) * 1e3;
        return parseFloat(str);
    }

    function sfParseMcap(str) {
        str = String(str || '').trim().toUpperCase().replace(/,/g,'');
        if (str.endsWith('B')) return parseFloat(str) * 1e9;
        if (str.endsWith('M')) return parseFloat(str) * 1e6;
        if (str.endsWith('K')) return parseFloat(str) * 1e3;
        return parseFloat(str);
    }

    function sfPillLabel(row) {
        var type = SF_TYPES.find(function(t){ return t.value === row.type; });
        var typeLabel = type ? type.label : row.type;
        if (row.type === 'ma') {
            var cond = row.maCondition || 'above_price';
            var mt = row.maType || 'SMA'; var mp = row.maPeriod || 50;
            var mt2 = row.maType2 || 'SMA'; var mp1 = row.maPeriod1 || 5; var mp2 = row.maPeriod2 || 50;
            if (cond === 'above_price')         return 'Price above ' + mt + ' ' + mp;
            if (cond === 'below_price')         return 'Price below ' + mt + ' ' + mp;
            if (cond === 'price_crossed_above') return 'Price crossed above ' + mt + ' ' + mp;
            if (cond === 'price_crossed_below') return 'Price crossed below ' + mt + ' ' + mp;
            if (cond === 'above_pct')     return mt + ' ' + mp + ' dist greater than ' + (row.val||0) + '%';
            if (cond === 'below_pct')     return mt + ' ' + mp + ' dist less than ' + (row.val||0) + '%';
            if (cond === 'between_pct')   return mt + ' ' + mp + ' dist between ' + (row.val||0) + '% and ' + (row.val2!=null?row.val2:10) + '%';
            if (cond === 'ma_above')      return mt + ' ' + mp1 + ' above ' + mt2 + ' ' + mp2;
            if (cond === 'ma_below')      return mt + ' ' + mp1 + ' below ' + mt2 + ' ' + mp2;
            if (cond === 'crosses_above') return mt + ' ' + mp1 + ' crossed above ' + mt2 + ' ' + mp2;
            if (cond === 'crosses_below') return mt + ' ' + mp1 + ' crossed below ' + mt2 + ' ' + mp2;
            if (cond === 'ma_cluster') {
                var cms = row.clusterMAs || ['SMA50','SMA200','EMA21'];
                return 'Cluster ' + cms.join('/') + ' ≤' + (row.clusterSpread!=null?row.clusterSpread:1) + '%';
            }
            if (cond === 'slope') {
                var sd = row.slopeDir || 'rising';
                var sdLabel = sd === 'rising' ? '↑' : sd === 'falling' ? '↓' : '—';
                var sdSuffix = sd === 'flat' ? '±' : '≥';
                return mt + ' ' + mp + ' Slope ' + sdLabel + ' ' + sdSuffix + (row.val!=null?row.val:(sd==='flat'?0.5:1)) + '%';
            }
            return 'MA filter';
        }
        if (row.type === 'perf') {
            var perfTfLabels = { '1d': '1D', '1w': '1W', '1m': '1M', '3m': '3M' };
            var perfTf = perfTfLabels[row.perfTf || '1d'] || '1D';
            var perfDir = row.perfDir === 'down' ? '↓' : '↑';
            var perfVal = row.val != null && row.val !== '' ? row.val : '0';
            return 'Perf ' + perfTf + ' ' + perfDir + ' ≥ ' + perfVal + '%';
        }
        if (row.type === 'pattern') {
            var patLabels = { inside_day: 'Inside Day', double_inside_day: 'Double Inside Day', bullish_outside: 'Bullish Outside', bearish_outside: 'Bearish Outside', hammer: 'Hammer', bullish_reversal_bar: 'Bullish Reversal Bar', upside_reversal: 'Upside Reversal', oops_reversal: 'Oops Reversal', pocket_pivot: 'Pocket Pivot' };
            var ptfLabel = { d: 'D', w: 'W', m: 'M' }[row.patternTf || 'd'];
            return (patLabels[row.val] || row.val) + ' · ' + ptfLabel;
        }
        if (row.type === 'fund') {
            var m = FUND_METRICS.find(function(x){ return x.value === row.fundMetric; });
            var mLabel = m ? m.label : (row.fundMetric || 'metric');
            var dirLabel2 = row.dir === 'gt' ? '>' : '<';
            return mLabel + ' ' + dirLabel2 + ' ' + (row.val !== '' && row.val != null ? row.val : '—') + '%';
        }
        if (row.type === 'valuation') {
            var vm = VAL_METRICS.find(function(x){ return x.value === row.valMetric; });
            var vmLabel = vm ? vm.label : (row.valMetric || 'metric');
            var vDir = row.dir === 'gt' ? '>' : '<';
            return vmLabel + ' ' + vDir + ' ' + (row.val !== '' && row.val != null ? row.val : '—');
        }
        if (row.type === 'gap') {
            var gapDir = row.dir === 'up' ? 'Up' : 'Down';
            return 'Gap ' + gapDir + ' ≥ ' + (row.val != null ? row.val : 0) + '%';
        }
        if (row.type === 'range') {
            if (row.rangeCondition === 'nr') {
                return 'NR' + (row.val != null ? row.val : '?');
            }
            return 'Range ≤ ' + (row.val != null ? row.val : '?') + '% of ADR';
        }
        if (row.type === 'rs') {
            var rsMetricLabel = (row.rsMetric === 'weighted_rs_pct') ? '3M RS' : 'RS';
            var rsDir = row.dir === 'gt' ? '>' : '<';
            return rsMetricLabel + ' ' + rsDir + ' ' + (row.val != null ? row.val : '—');
        }
        if (row.type === 'cr') {
            var crTfLabel = { d: 'D', w: 'W', m: 'M' }[row.crTf || 'd'];
            var crDir = row.dir === 'gt' ? '>' : '<';
            return 'CR ' + crTfLabel + ' ' + crDir + ' ' + (row.val != null ? row.val : '—') + '%';
        }
        if (row.type === 'wk52') {
            var side    = row.wk52Side === 'low' ? '52W Low' : '52W High';
            var newOnly = row.wk52NewOnly ? ' · New' : '';
            var hasMin  = row.wk52DistMin != null && row.wk52DistMin !== '';
            var hasMax  = row.wk52DistMax != null && row.wk52DistMax !== '';
            var dVal    = '';
            if (!row.wk52NewOnly) {
                if (hasMin && hasMax) dVal = ' ' + row.wk52DistMin + ' – ' + row.wk52DistMax + '%';
                else if (hasMin)     dVal = ' ≥' + row.wk52DistMin + '%';
                else if (hasMax)     dVal = ' ≤' + row.wk52DistMax + '%';
            }
            return side + newOnly + dVal;
        }
        if (row.type === 'indrank') {
            var irMode = row.indrankMode || 'top';
            var irModeLabel = irMode === 'top' ? 'Top' : 'Below';
            return 'Ind Rank ' + irModeLabel + ' ' + (row.val != null && row.val !== '' ? row.val : '—');
        }
        if (row.type === 'rsi') {
            var rsiDir = row.dir === 'gt' ? '>' : '<';
            return 'RSI ' + rsiDir + ' ' + (row.val != null ? row.val : '—');
        }
        if (row.type === 'sector') {
            var secs = row.sectors || [];
            var inds = row.industries || [];
            if (inds.length > 0) return inds.length <= 2 ? inds.join(', ') : inds[0] + ' +' + (inds.length - 1);
            if (secs.length > 0) return secs.length <= 2 ? secs.join(', ') : secs[0] + ' +' + (secs.length - 1);
            return 'Sector';
        }
        var dirLabel = row.dir === 'gt' ? 'greater than' : 'less than';
        var valLabel = row.val !== '' && row.val != null ? ' ' + row.val : '';
        if (row.type === 'adr') valLabel += '%';
        if (row.type === 'rvol') valLabel += 'x';
        if (row.type === 'udv') valLabel = ' (' + (row.udvPeriod || 50) + 'D) >' + (row.val !== '' && row.val != null ? ' ' + row.val : '');
        if (row.type === 'udv') return 'U/D Vol' + valLabel;
        return typeLabel + ' ' + dirLabel + valLabel;
    }

    // Map type → current active row (if any)
    function sfRowForType(type) {
        return sfRows.find(function(r){ return r.type === type; }) || null;
    }

    function sfPillValueLabel(row) {
        if (!row) return '';
        return sfPillLabel(row);
    }

    function sfRenderPills() {
        var bar = document.getElementById('sf-pill-bar');
        if (!bar) return;

        var PILL_TYPES = [
            { type: 'rs',      label: 'RS' },
            { type: 'price',   label: 'Price' },
            { type: 'vol',     label: 'Avg Vol' },
            { type: 'adr',     label: 'ADR%' },
            { type: 'cr',      label: 'CR%' },
            { type: 'mcap',    label: 'Mkt Cap' },
            { type: 'rvol',    label: 'Rel. Vol' },
            { type: 'pattern', label: 'Pattern' },
            { type: 'ma',      label: 'MA' },
            { type: 'fund',    label: 'Fundamental' },
        ];

        var SF_TOOLTIPS = {
            rs:      'Relative Strength percentile vs all stocks (1–99). Higher = stronger price action.',
            price:   'Filter by current stock price.',
            vol:     'Average daily trading volume over 50 days.',
            adr:     'Average Daily Range % — measures a stock\'s typical daily price volatility.',
            cr:      'Closing Range % — where price closed within the day\'s high/low. 100% = closed at the high.',
            mcap:    'Market Cap — total market value of the company.',
            rvol:    'Relative Volume — today\'s volume vs the average for this time of day.',
            pattern: 'Filter by candlestick or price pattern.',
            ma:      'Moving Average — filter by price position relative to a moving average.',
            fund:    'Fundamental metrics — EPS growth, sales growth, margins and more.',
            valuation: 'Valuation ratios — Forward P/E, P/S, and PEG.',
            udv:     'Up/Down Volume ratio — compares volume on up days vs down days over a selected period.',
            perf:    'Price performance over a selected timeframe.',
            gap:     'Filter stocks that gapped up or down at the open.',
            range:   'Filter by where price sits within its 52-week or recent range.',
            sector:  'Filter by market sector or specific industry.',
            wk52:    'Filter by 52-week high or low — new highs/lows and distance from each.',
            indrank: 'Filter by industry rank — show only stocks in top-ranked or lower-ranked industries.',
            rsi:     'RSI — Relative Strength Index over 14 days. Filter above or below a threshold (1–99).',
        };

        var html = PILL_TYPES.map(function(pt) {
            var row   = sfRowForType(pt.type);
            var isOpen   = sfActivePillType === pt.type;
            var isActive = !!row;
            var cls   = 'sf-pill' + (isActive ? ' active' : '') + (isOpen ? ' open' : '');
            var val   = isActive ? sfPillValueLabel(row) : '';
            var tip   = SF_TOOLTIPS[pt.type] || '';
            var valHtml = isActive
                ? '<span class="sf-pill-val">' + esc(val.replace(pt.label + ' ', '').replace(pt.label, '').trim()) + '</span>'
                : '';
            var xBtn = isActive
                ? '<button class="sf-pill-x" onclick="event.stopPropagation();sfRemovePillType(\'' + pt.type + '\')">✕</button>'
                : '';
            return '<div class="' + cls + '" data-rs-tip="' + tip + '" onclick="sfTogglePill(\'' + pt.type + '\',this,event)">' +
                '<span class="sf-pill-name">' + pt.label + '</span>' +
                valHtml +
                '<span class="sf-pill-caret">' + (isOpen ? '▴' : '▾') + '</span>' +
                xBtn +
            '</div>';
        }).join('');

        var fixedTypes = PILL_TYPES.map(function(pt){ return pt.type; });

        // Render any extra rows added via + Filter (duplicates of existing types, or non-fixed types)
        var extraRows = sfRows.filter(function(r) {
            if (fixedTypes.indexOf(r.type) === -1) return true; // non-fixed type (e.g. gap)
            var sameType = sfRows.filter(function(r2){ return r2.type === r.type; });
            return sameType.length > 1 && sameType.indexOf(r) > 0;
        });
        var extraHtml = extraRows.map(function(row) {
            var val = sfPillLabel(row);
            var tip = SF_TOOLTIPS[row.type] || '';
            return '<div class="sf-pill active" data-rs-tip="' + tip + '" onclick="sfOpenPopover(' + row.id + ',this)">' +
                '<span class="sf-pill-val" style="padding-left:8px;">' + esc(val) + '</span>' +
                '<span class="sf-pill-caret">▾</span>' +
                '<button class="sf-pill-x" onclick="event.stopPropagation();sfRemoveRow(' + row.id + ')">✕</button>' +
            '</div>';
        }).join('');

        html += extraHtml;

        // + Filter button for adding additional filters
        html += '<button class="sf-pill-add" onclick="sfAddRow()">+ Filter</button>';

        bar.innerHTML = html;
    }

    function sfPopoverHtml(row, noTypeRow) {
        var id = row.id;
        var html = '';
        if (!noTypeRow) {
            var typeOpts = SF_TYPES.map(function(t){
                return '<option value="'+t.value+'"'+(row.type===t.value?' selected':'')+'>'+t.label+'</option>';
            }).join('');
            html = '<div class="sf-popover-row">' +
                '<span class="sf-popover-label">Type</span>' +
                '<select class="sf-select" onchange="sfPopChange('+id+',\'type\',this.value)">'+typeOpts+'</select>' +
                '</div>';
        }
        if (row.type === 'ma') {
            var MA_P = ['5','8','10','21','50','65','150','200'];
            var cond = row.maCondition || 'above_price';
            var condOpts = [
                ['above_price',         'Price Above'],
                ['below_price',         'Price Below'],
                ['price_crossed_above', 'Price Crossed Above'],
                ['price_crossed_below', 'Price Crossed Below'],
                ['above_pct',           'Above % (Dist)'],
                ['below_pct',           'Below % (Dist)'],
                ['between_pct',         'Between % (Dist)'],
                ['ma_above',            'MA Above MA'],
                ['ma_below',            'MA Below MA'],
                ['crosses_above',       'Crosses Above'],
                ['crosses_below',       'Crosses Below'],
                ['ma_cluster',          'MA Cluster'],
                ['slope',               'Slope'],
            ].map(function(o){ return '<option value="'+o[0]+'"'+(cond===o[0]?' selected':'')+'>'+o[1]+'</option>'; }).join('');
            html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="How price relates to the moving average. Crossed conditions only trigger on today\'s candle.">Condition</span><select class="sf-select" onchange="sfPopChange('+id+',\'maCondition\',this.value)">'+condOpts+'</select></div>';
            // Single MA pickers (price above/below, dist)
            var usesSingleMA = ['above_price','below_price','price_crossed_above','price_crossed_below','above_pct','below_pct','between_pct','slope'].indexOf(cond) !== -1;
            var usesDualMA   = ['ma_above','ma_below','crosses_above','crosses_below'].indexOf(cond) !== -1;
            var usesPct      = ['above_pct','below_pct'].indexOf(cond) !== -1;
            var usesBetween  = cond === 'between_pct';
            var mt  = ['SMA','EMA'].map(function(t){ return '<option value="'+t+'"'+(row.maType===t?' selected':'')+'>'+t+'</option>'; }).join('');
            var mp  = MA_P.map(function(p){ return '<option value="'+p+'"'+(String(row.maPeriod)===p?' selected':'')+'>'+p+'</option>'; }).join('');
            var mt2 = ['SMA','EMA'].map(function(t){ return '<option value="'+t+'"'+(row.maType2===t?' selected':'')+'>'+t+'</option>'; }).join('');
            var mp1 = MA_P.map(function(p){ return '<option value="'+p+'"'+(String(row.maPeriod1)===p?' selected':'')+'>'+p+'</option>'; }).join('');
            var mp2 = MA_P.map(function(p){ return '<option value="'+p+'"'+(String(row.maPeriod2)===p?' selected':'')+'>'+p+'</option>'; }).join('');
            if (usesSingleMA) {
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="SMA weights all days equally. EMA weights recent days more heavily.">MA Type</span><select class="sf-select" onchange="sfPopChange('+id+',\'maType\',this.value)">'+mt+'</select></div>';
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Number of days in the moving average calculation.">Period</span><select class="sf-select" onchange="sfPopChange('+id+',\'maPeriod\',this.value)">'+mp+'</select></div>';
            }
            if (usesDualMA) {
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="The shorter, more responsive moving average.">Fast MA</span><select class="sf-select" style="width:70px;" onchange="sfPopChange('+id+',\'maType\',this.value)">'+mt+'</select><select class="sf-select" style="width:70px;margin-left:4px;" onchange="sfPopChange('+id+',\'maPeriod1\',this.value)">'+mp1+'</select></div>';
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="The longer, slower-moving average used as the trend baseline.">Slow MA</span><select class="sf-select" style="width:70px;" onchange="sfPopChange('+id+',\'maType2\',this.value)">'+mt2+'</select><select class="sf-select" style="width:70px;margin-left:4px;" onchange="sfPopChange('+id+',\'maPeriod2\',this.value)">'+mp2+'</select></div>';
            }
            if (usesPct) {
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Minimum % distance between price and the moving average.">Value</span><input class="sf-input" type="number" step="0.1" value="'+(row.val||0)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
            }
            if (usesBetween) {
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Lower bound: minimum % the price must be below the MA.">Min</span><input class="sf-input" type="number" step="0.1" value="'+(row.val||0)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Upper bound: maximum % the price can be above the MA.">Max</span><input class="sf-input" type="number" step="0.1" value="'+(row.val2!=null?row.val2:10)+'" oninput="sfPopChange('+id+',\'val2\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
            }
            if (cond === 'slope') {
                var slopeDir = row.slopeDir || 'rising';
                var slopeBtns = [['rising','↑ Rising'],['falling','↓ Falling'],['flat','— Flat']].map(function(s){
                    return '<button class="sf-seg-btn'+(slopeDir===s[0]?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'slopeDir\',\''+s[0]+'\')">'+s[1]+'</button>';
                }).join('');
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Whether the MA is angling up, down, or staying flat.">Direction</span><div class="sf-seg">'+slopeBtns+'</div></div>';
                if (slopeDir === 'flat') {
                    html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Maximum % change per day for the slope to still qualify as flat.">Max ±</span><input class="sf-input" type="number" step="0.1" min="0" value="'+(row.val!=null?row.val:0.5)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
                } else {
                    html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Minimum slope steepness required, as % change per day.">Min %</span><input class="sf-input" type="number" step="0.1" value="'+(row.val!=null?row.val:1)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
                }
            }
            if (cond === 'ma_cluster') {
                var clusterMAs = [
                    ['SMA','5'],['EMA','8'],['EMA','21'],
                    ['SMA','50'],['EMA','65'],['SMA','150'],['SMA','200'],
                ];
                var selectedMAs = row.clusterMAs || ['SMA50','EMA21','SMA200'];
                var checks = clusterMAs.map(function(m) {
                    var key = m[0]+m[1];
                    var on  = selectedMAs.indexOf(key) !== -1;
                    return '<span class="sf-cluster-chip' + (on ? ' on' : '') + '" data-key="' + key + '" data-id="' + id + '" onclick="event.stopPropagation();sfToggleClusterMA(this)">' + m[0] + ' ' + m[1] + '</span>';
                }).join('');
                html += '<div class="sf-popover-row" style="flex-wrap:wrap;gap:3px;"><span class="sf-popover-label" style="width:100%;margin-bottom:2px;" data-rs-tip="Select which moving averages must be grouped tightly together.">MAs</span>'+checks+'</div>';
                html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Maximum % spread allowed between the highest and lowest selected MA.">Max spread</span><input class="sf-input" type="number" step="0.1" min="0.1" value="'+(row.clusterSpread!=null?row.clusterSpread:1)+'" oninput="sfPopChange('+id+',\'clusterSpread\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
            }
        } else if (row.type === 'pattern') {
            var patOpts = [
                ['inside_day',          'Inside Day'],
                ['double_inside_day',   'Double Inside Day'],
                ['bullish_outside',     'Bullish Outside'],
                ['bearish_outside',     'Bearish Outside'],
                ['hammer',              'Hammer'],
                ['bullish_reversal_bar','Bullish Reversal Bar'],
                ['upside_reversal',     'Upside Reversal'],
                ['oops_reversal',       'Oops Reversal'],
                ['pocket_pivot',        'Pocket Pivot'],
            ].map(function(o){ return '<option value="'+o[0]+'"'+(row.val===o[0]?' selected':'')+'>'+o[1]+'</option>'; }).join('');
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Pattern</span><select class="sf-select" onchange="sfPopChange('+id+',\'val\',this.value)">'+patOpts+'</select></div>';
            var ptf = row.patternTf || 'd';
            var ptfBtns = [['d','D'],['w','W'],['m','M']].map(function(t){
                return '<button class="sf-seg-btn'+(ptf===t[0]?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'patternTf\',\''+t[0]+'\')">'+t[1]+'</button>';
            }).join('');
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Timeframe</span><div class="sf-seg">'+ptfBtns+'</div></div>';
        } else if (row.type === 'fund') {
            var metricOpts = FUND_METRICS.map(function(m){
                return '<option value="'+m.value+'"'+(row.fundMetric===m.value?' selected':'')+'>'+m.label+'</option>';
            }).join('');
            var dirOpts3 = '<option value="gt"'+(row.dir==='gt'?' selected':'')+'>Greater than</option><option value="lt"'+(row.dir==='lt'?' selected':'')+'>Less than</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Metric</span><select class="sf-select" style="max-width:160px;" onchange="sfPopChange('+id+',\'fundMetric\',this.value)">'+metricOpts+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Condition</span><select class="sf-select" onchange="sfPopChange('+id+',\'dir\',this.value)">'+dirOpts3+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Value</span><input class="sf-input" type="number" step="0.1" value="'+(row.val!=null?row.val:0)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
        } else if (row.type === 'valuation') {
            var valMetricOpts = VAL_METRICS.map(function(m){
                return '<option value="'+m.value+'"'+(row.valMetric===m.value?' selected':'')+'>'+m.label+'</option>';
            }).join('');
            var valDirOpts = '<option value="gt"'+(row.dir==='gt'?' selected':'')+'>Above</option><option value="lt"'+(row.dir==='lt'?' selected':'')+'>Below</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Metric</span><select class="sf-select" style="max-width:160px;" onchange="sfPopChange('+id+',\'valMetric\',this.value)">'+valMetricOpts+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Condition</span><select class="sf-select" onchange="sfPopChange('+id+',\'dir\',this.value)">'+valDirOpts+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Value</span><input class="sf-input" type="number" step="0.1" min="0" value="'+(row.val!=null?row.val:20)+'" oninput="sfPopChange('+id+',\'val\',this.value)"></div>';
        } else if (row.type === 'gap') {
            var gapDirOpts = '<option value="up"'+(row.dir==='up'?' selected':'')+'>Gap Up</option><option value="down"'+(row.dir==='down'?' selected':'')+'>Gap Down</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Direction</span><select class="sf-select" onchange="sfPopChange('+id+',\'dir\',this.value)">'+gapDirOpts+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Min %</span><input class="sf-input" type="number" min="0" step="0.1" value="'+(row.val!=null?row.val:1)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
        } else if (row.type === 'perf') {
            var perfTf = row.perfTf || '1d';
            var perfDir = row.perfDir || 'up';
            var tfs = [['1d','1D'],['1w','1W'],['1m','1M'],['3m','3M']];
            var tfBtns = tfs.map(function(t){
                return '<button class="sf-seg-btn'+(perfTf===t[0]?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'perfTf\',\''+t[0]+'\')">'+t[1]+'</button>';
            }).join('');
            var dirBtns =
                '<button class="sf-seg-btn'+(perfDir==='up'?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'perfDir\',\'up\')">↑ Up</button>' +
                '<button class="sf-seg-btn'+(perfDir==='down'?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'perfDir\',\'down\')">↓ Down</button>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Direction</span><div class="sf-seg">'+dirBtns+'</div></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Timeframe</span><div class="sf-seg">'+tfBtns+'</div></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Min %</span><input class="sf-input" type="number" step="0.1" value="'+(row.val!=null?row.val:5)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
        } else if (row.type === 'range') {
            var rc = row.rangeCondition || 'relative';
            var rcOpts = '<option value="relative"'+(rc==='relative'?' selected':'')+'>Relative Range</option><option value="nr"'+(rc==='nr'?' selected':'')+'>Narrow Range (NR)</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Condition</span><select class="sf-select" onchange="sfPopChange('+id+',\'rangeCondition\',this.value)">'+rcOpts+'</select></div>';
            if (rc === 'relative') {
                html += '<div class="sf-popover-row"><span class="sf-popover-label">Max %</span><input class="sf-input" type="number" min="1" max="200" step="1" value="'+(row.val!=null?row.val:'')+'" placeholder="e.g. 50" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">% of ADR</span></div>';
            } else {
                html += '<div class="sf-popover-row"><span class="sf-popover-label">Days</span><input class="sf-input" type="number" min="2" max="60" step="1" value="'+(row.val!=null?row.val:'')+'" placeholder="e.g. 7" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">days</span></div>';
            }
        } else if (row.type === 'cr') {
            var crTf = row.crTf || 'd';
            var crTfBtns = [['d','D'],['w','W'],['m','M']].map(function(t){
                return '<button class="sf-seg-btn'+(crTf===t[0]?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'crTf\',\''+t[0]+'\')">'+t[1]+'</button>';
            }).join('');
            var crDirOpts = '<option value="gt"'+(row.dir==='gt'?' selected':'')+'>Greater than</option><option value="lt"'+(row.dir==='lt'?' selected':'')+'>Less than</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Direction</span><select class="sf-select" onchange="sfPopChange('+id+',\'dir\',this.value)">'+crDirOpts+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Value</span><input class="sf-input" type="number" min="0" max="100" step="1" value="'+(row.val!=null?row.val:50)+'" oninput="sfPopChange('+id+',\'val\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Timeframe</span><div class="sf-seg">'+crTfBtns+'</div></div>';
        } else if (row.type === 'rs') {
            var rsMetric = row.rsMetric || 'Percentile';
            var rsMetricOpts = '<option value="Percentile"'+(rsMetric==='Percentile'?' selected':'')+'>RS</option>' +
                               '<option value="weighted_rs_pct"'+(rsMetric==='weighted_rs_pct'?' selected':'')+'>3M RS</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Metric</span><select class="sf-select" onchange="sfPopChange('+id+',\'rsMetric\',this.value)">'+rsMetricOpts+'</select></div>';
            var dirOptsRs = '<option value="gt"'+(row.dir==='gt'?' selected':'')+'>Greater than</option><option value="lt"'+(row.dir==='lt'?' selected':'')+'>Less than</option>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Direction</span><select class="sf-select" onchange="sfPopChange('+id+',\'dir\',this.value)">'+dirOptsRs+'</select></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Value</span><input class="sf-input" type="number" min="1" max="99" step="1" value="'+(row.val!=null?row.val:70)+'" oninput="sfPopChange('+id+',\'val\',this.value)"></div>';
        } else if (row.type === 'wk52') {
            var wk52Side    = row.wk52Side    || 'high';
            var wk52NewOnly = row.wk52NewOnly || 0;
            var wk52DistMin = row.wk52DistMin != null ? row.wk52DistMin : '';
            var wk52DistMax = row.wk52DistMax != null ? row.wk52DistMax : '';
            // Mutually exclusive chips: High / Low
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Side</span>' +
                '<div class="sf-seg">' +
                '<button class="sf-seg-btn'+(wk52Side==='high'?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'wk52Side\',\'high\')">52W High</button>' +
                '<button class="sf-seg-btn'+(wk52Side==='low'?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'wk52Side\',\'low\')">52W Low</button>' +
                '</div></div>';
            // New only toggle — No = show distance rows, Yes = hide them
            html += '<div class="sf-popover-row"><span class="sf-popover-label">New only</span>' +
                '<div class="sf-seg">' +
                '<button class="sf-seg-btn'+(wk52NewOnly?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'wk52NewOnly\',1)">Yes</button>' +
                '<button class="sf-seg-btn'+(!wk52NewOnly?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'wk52NewOnly\',0)">No</button>' +
                '</div></div>';
            // Distance rows only shown when New Only = No
            if (!wk52NewOnly) {
                html += '<div class="sf-popover-row"><span class="sf-popover-label">From %</span><input class="sf-input" type="number" min="0" step="0.1" placeholder="e.g. 5" value="'+wk52DistMin+'" oninput="sfPopChange('+id+',\'wk52DistMin\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
                html += '<div class="sf-popover-row"><span class="sf-popover-label">To %</span><input class="sf-input" type="number" min="0" step="0.1" placeholder="e.g. 10" value="'+wk52DistMax+'" oninput="sfPopChange('+id+',\'wk52DistMax\',this.value)"><span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span></div>';
            }
        } else if (row.type === 'sector') {
            var allSectors = [];
            var sectorIndustries = {};
            if (industriesData && industriesData.industries) {
                industriesData.industries.forEach(function(ind) {
                    if (allSectors.indexOf(ind.sector) === -1) {
                        allSectors.push(ind.sector);
                        sectorIndustries[ind.sector] = [];
                    }
                    sectorIndustries[ind.sector].push(ind.industry);
                });
                allSectors.sort();
                allSectors.forEach(function(s){ sectorIndustries[s].sort(); });
            }
            var selSectors = row.sectors || [];
            var selIndustries = row.industries || [];
            var sectorChips = allSectors.map(function(s) {
                var on = selSectors.indexOf(s) !== -1;
                return '<span class="sf-cluster-chip'+(on?' on':'')+'" data-val="'+esc(s)+'" data-id="'+id+'" onclick="event.stopPropagation();sfToggleSectorChip(this)">'+esc(s)+'</span>';
            }).join('');
            var visibleIndustries = [];
            if (selSectors.length > 0) {
                selSectors.forEach(function(s){ if (sectorIndustries[s]) visibleIndustries = visibleIndustries.concat(sectorIndustries[s]); });
                visibleIndustries.sort();
            } else {
                allSectors.forEach(function(s){ visibleIndustries = visibleIndustries.concat(sectorIndustries[s] || []); });
                visibleIndustries.sort();
            }
            var industryChips = visibleIndustries.map(function(ind) {
                var on = selIndustries.indexOf(ind) !== -1;
                return '<span class="sf-cluster-chip'+(on?' on':'')+'" data-val="'+esc(ind)+'" data-id="'+id+'" onclick="event.stopPropagation();sfToggleIndustryChip(this)">'+esc(ind)+'</span>';
            }).join('');
            html += '<div class="sf-popover-row" style="flex-direction:column;align-items:flex-start;"><span class="sf-popover-label" style="width:100%;margin-bottom:4px;">Sector</span><div style="display:flex;flex-wrap:wrap;gap:3px;">'+sectorChips+'</div></div>';
            html += '<div class="sf-popover-row" style="flex-direction:column;align-items:flex-start;margin-top:4px;"><span class="sf-popover-label" style="width:100%;margin-bottom:4px;">Industry</span><div style="display:flex;flex-wrap:wrap;gap:3px;max-height:160px;overflow-y:auto;padding-right:2px;">'+(industryChips||'<span style="color:#484f58;font-size:0.748em;">No industry data</span>')+'</div></div>';
        } else if (row.type === 'indrank') {
            var irMode = row.indrankMode || 'top';
            var irBtns = [['top','Top'],['below','Below']].map(function(m){
                return '<button class="sf-seg-btn'+(irMode===m[0]?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'indrankMode\',\''+m[0]+'\')">'+m[1]+'</button>';
            }).join('');
            html += '<div class="sf-popover-row"><span class="sf-popover-label" data-rs-tip="Top N: stocks in industries ranked 1 through N. Below N: stocks in industries ranked lower than N.">Mode</span><div class="sf-seg">'+irBtns+'</div></div>';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Rank</span><input class="sf-input" type="number" min="1" step="1" value="'+(row.val!=null&&row.val!==''?row.val:15)+'" oninput="sfPopChange('+id+',\'val\',this.value)"></div>';
        } else {
            var dirOpts2 = '<option value="gt"'+(row.dir==='gt'?' selected':'')+'>Greater than</option><option value="lt"'+(row.dir==='lt'?' selected':'')+'>Less than</option>';
            if (row.type !== 'udv') {
                html += '<div class="sf-popover-row"><span class="sf-popover-label">Direction</span><select class="sf-select" onchange="sfPopChange('+id+',\'dir\',this.value)">'+dirOpts2+'</select></div>';
            }
            if (row.type === 'udv') {
                var p20 = (row.udvPeriod || 50) === 20;
                html += '<div class="sf-popover-row"><span class="sf-popover-label">Period</span>' +
                    '<div class="sf-seg">' +
                    '<button class="sf-seg-btn'+(p20?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'udvPeriod\',20)">20D</button>' +
                    '<button class="sf-seg-btn'+(!p20?' active':'')+'" onclick="event.stopPropagation();sfPopChange('+id+',\'udvPeriod\',50)">50D</button>' +
                    '</div></div>';
            }
            var inputType = (row.type === 'vol' || row.type === 'mcap') ? 'text' : 'number';
            var minMax = row.type === 'rvol' ? ' min="0.1" max="20" step="0.1"' : row.type === 'udv' ? ' min="0.1" max="20" step="0.1"' : row.type === 'adr' ? ' min="0" step="0.1"' : row.type === 'price' ? ' min="0" step="0.01"' : '';
            var suffix = row.type === 'adr' ? '<span style="color:#6e7681;font-size:0.748em;margin-left:2px;">%</span>' : row.type === 'rvol' ? '<span style="color:#6e7681;font-size:0.748em;margin-left:2px;">x</span>' : '';
            html += '<div class="sf-popover-row"><span class="sf-popover-label">Value</span><input class="sf-input" type="'+inputType+'"'+minMax+' value="'+row.val+'" oninput="sfPopChange('+id+',\'val\',this.value)">'+suffix+'</div>';
        }
        return html;
    }

    var _sfJustClosed = false;

    window.sfTogglePill = function(type, pillEl, e) {
        if (e) e.stopPropagation();
        if (_sfJustClosed) { _sfJustClosed = false; return; }
        if (sfActivePillType === type) {
            _sfJustClosed = true;
            // If the row was freshly created (never applied), remove it — same as cancel
            if (!sfPillWasPreExisting) {
                sfRows = sfRows.filter(function(r){ return r.type !== type; });
            }
            document.getElementById('sf-popover').style.display = 'none';
            sfActivePillType = null;
            sfPillWasPreExisting = false;
            sfRenderPills();
            // Don't call renderScans — nothing was applied
            setTimeout(function(){ _sfJustClosed = false; }, 200);
            return;
        }
        sfActivePillType = type;

        // Ensure a row exists for this type
        var row = sfRowForType(type);
        sfPillWasPreExisting = !!row;
        if (!row) {
            sfRowId++;
            var defaults = { id: sfRowId, type: type, dir: 'gt',
                maCondition: 'above_price', maType: 'SMA', maPeriod: 50,
                maType2: 'SMA', maPeriod1: 5, maPeriod2: 50,
                fundMetric: type === 'fund' ? 'eps_next_y_pct' : null,
                rsMetric: 'Percentile',
                perfTf: type === 'perf' ? '1d' : null,
                perfDir: type === 'perf' ? 'up' : null,
                patternTf: 'd',
                crTf: 'd',
                udvPeriod: 50,
                sectors: type === 'sector' ? [] : undefined,
                industries: type === 'sector' ? [] : undefined,
                indrankMode: type === 'indrank' ? 'top' : undefined,
                val: type === 'rs' ? 70 : type === 'price' ? 5 : type === 'vol' ? '500K' :
                     type === 'adr' ? 3 : type === 'cr' ? 50 : type === 'mcap' ? '1B' :
                     type === 'rvol' ? 1 : type === 'udv' ? 1.5 : type === 'pattern' ? 'inside_day' : type === 'gap' ? 1 : type === 'range' ? 50 : type === 'sector' ? null : type === 'indrank' ? 15 : 0,
                dir: type === 'gap' ? 'up' : 'gt' };
            sfRows.push(defaults);
            row = defaults;
        }

        var TITLES = { rs: 'RS', price: 'Price', vol: 'Avg Volume',
                       adr: 'ADR%', cr: 'Closing Range %', mcap: 'Market Cap',
                       rvol: 'Relative Volume', udv: 'U/D Vol', pattern: 'Pattern', ma: 'Moving Average',
                       fund: 'Fundamental', valuation: 'Valuation', gap: 'Gap', perf: 'Performance', sector: 'Sector / Industry',
                       indrank: 'Industry Rank' };
        document.getElementById('sf-popover-title').textContent = TITLES[type] || type;
        var pop  = document.getElementById('sf-popover');
        pop.style.maxWidth = (type === 'sector') ? '380px' : '260px';

        // Build content — type row omitted since type is fixed per pill
        var content = sfPopoverHtml(row, true);
        document.getElementById('sf-popover-content').innerHTML = content;

        var pop  = document.getElementById('sf-popover');
        var rect = pillEl.getBoundingClientRect();
        var top  = rect.bottom + 4;
        var left = Math.min(rect.left, window.innerWidth - 260);
        if (top + 260 > window.innerHeight) top = rect.top - 270;
        pop.style.top  = top + 'px';
        pop.style.left = left + 'px';
        pop.style.display = 'block';
    };

    window.sfClearPillType = function(type) {
        var row = sfRowForType(type);
        if (row) {
            // Reset to defaults for this type
            row.dir         = 'gt';
            row.maCondition = 'above_price';
            row.maType      = 'SMA'; row.maPeriod  = 50;
            row.maType2     = 'SMA'; row.maPeriod1 = 5; row.maPeriod2 = 50;
            row.fundMetric  = type === 'fund' ? 'eps_next_y_pct' : null;
            row.rsMetric    = 'Percentile';
            row.perfTf      = '1d';
            row.perfDir     = 'up';
            row.patternTf   = 'd';
            row.crTf        = 'd';
            row.sectors     = [];
            row.industries  = [];
            row.indrankMode = type === 'indrank' ? 'top' : row.indrankMode;
            row.val         = type === 'rs' ? 70 : type === 'price' ? 5 : type === 'vol' ? '500K' :
                              type === 'adr' ? 3 : type === 'cr' ? 50 : type === 'mcap' ? '1B' :
                              type === 'rvol' ? 1 : type === 'udv' ? 1.5 : type === 'pattern' ? 'inside_day' : type === 'perf' ? 5 : type === 'sector' ? null : type === 'indrank' ? 15 : 0;
            row.val2        = null;
            // Re-render popover content with reset values
            var c = sfPopoverHtml(row, true);
            var contentEl = document.getElementById('sf-popover-content');
            if (contentEl) contentEl.innerHTML = c;
        }
    };

    window.sfResetPill = function() {
        if (sfActivePillType) {
            sfClearPillType(sfActivePillType);
        } else if (sfActivePopoverId != null) {
            var row = sfRows.find(function(r){ return r.id === sfActivePopoverId; });
            if (!row) return;
            row.perfTf  = '1d';
            row.perfDir = 'up';
            row.sectors    = [];
            row.industries = [];
            row.indrankMode = row.type === 'indrank' ? 'top' : row.indrankMode;
            row.val     = row.type === 'perf' ? 5 : row.type === 'gap' ? 1 : row.type === 'range' ? 50 : row.type === 'sector' ? null : row.type === 'indrank' ? 15 : 0;
            row.val2    = null;
            row.rangeCondition = row.type === 'range' ? 'relative' : null;
            var c = sfPopoverHtml(row, true);
            var contentEl = document.getElementById('sf-popover-content');
            if (contentEl) contentEl.innerHTML = c;
        }
    };

    window.sfRemovePillType = function(type) {
        sfRows = sfRows.filter(function(r){ return r.type !== type; });
        if (sfActivePillType === type) {
            document.getElementById('sf-popover').style.display = 'none';
            sfActivePillType = null;
            sfPillWasPreExisting = false;
        }
        sfRenderPills();
        renderScans();
    };

    window.sfCancelPill = function() {
        // If row was newly created (never applied), remove it
        if (!sfPillWasPreExisting && sfActivePillType) {
            sfRows = sfRows.filter(function(r){ return r.type !== sfActivePillType; });
        }
        // Also remove freshly-added extra rows (e.g. U/D Vol via + Filter)
        if (sfPopoverIsNew && sfActivePopoverId != null) {
            sfRows = sfRows.filter(function(r){ return r.id !== sfActivePopoverId; });
        }
        document.getElementById('sf-popover').style.display = 'none';
        sfActivePopoverId = null;
        sfActivePillType = null;
        sfPillWasPreExisting = false;
        sfPopoverIsNew = false;
        sfRenderPills();
        // Do NOT call renderScans — nothing changed
    };
    window.sfToggleSectorChip = function(el) {
        var id  = parseInt(el.getAttribute('data-id'));
        var val = el.getAttribute('data-val');
        var row = sfRows.find(function(r){ return r.id === id; });
        if (!row) return;
        if (!row.sectors)    row.sectors    = [];
        if (!row.industries) row.industries = [];
        var idx = row.sectors.indexOf(val);
        if (idx !== -1) { row.sectors.splice(idx, 1); }
        else { row.sectors.push(val); }
        // Drop any industry selections that no longer belong to the remaining sectors
        if (row.sectors.length > 0 && row.industries.length > 0) {
            row.industries = row.industries.filter(function(ind) {
                var indObj = industriesData && industriesData.industries &&
                    industriesData.industries.find(function(i){ return i.industry === ind; });
                return indObj && row.sectors.indexOf(indObj.sector) !== -1;
            });
        }
        var contentEl = document.getElementById('sf-popover-content');
        // Save scroll positions before rebuilding DOM (prevents browser from scrolling page on innerHTML replace)
        var tableWrap = document.querySelector('#scans-table-view .stocks-table-wrap');
        var tableScroll = tableWrap ? tableWrap.scrollTop : 0;
        var indDiv = contentEl ? contentEl.querySelector('div[style*="overflow-y"]') : null;
        var indScroll = indDiv ? indDiv.scrollTop : 0;
        contentEl.innerHTML = sfPopoverHtml(row, true);
        // Restore scroll positions
        if (tableWrap) tableWrap.scrollTop = tableScroll;
        var indDivNew = contentEl ? contentEl.querySelector('div[style*="overflow-y"]') : null;
        if (indDivNew) indDivNew.scrollTop = indScroll;
    };

    window.sfToggleIndustryChip = function(el) {
        var id  = parseInt(el.getAttribute('data-id'));
        var val = el.getAttribute('data-val');
        var row = sfRows.find(function(r){ return r.id === id; });
        if (!row) return;
        if (!row.industries) row.industries = [];
        var idx = row.industries.indexOf(val);
        if (idx !== -1) { row.industries.splice(idx, 1); }
        else { row.industries.push(val); }
        var contentEl = document.getElementById('sf-popover-content');
        var tableWrap = document.querySelector('#scans-table-view .stocks-table-wrap');
        var tableScroll = tableWrap ? tableWrap.scrollTop : 0;
        var indDiv = contentEl ? contentEl.querySelector('div[style*="overflow-y"]') : null;
        var indScroll = indDiv ? indDiv.scrollTop : 0;
        contentEl.innerHTML = sfPopoverHtml(row, true);
        if (tableWrap) tableWrap.scrollTop = tableScroll;
        var indDivNew = contentEl ? contentEl.querySelector('div[style*="overflow-y"]') : null;
        if (indDivNew) indDivNew.scrollTop = indScroll;
    };

    window.sfToggleClusterMA = function(el) {
        var id  = parseInt(el.getAttribute('data-id'));
        var key = el.getAttribute('data-key');
        var row = sfRows.find(function(r){ return r.id === id; });
        if (!row) return;
        if (!row.clusterMAs) row.clusterMAs = ['SMA50','EMA21','SMA200'];
        var idx = row.clusterMAs.indexOf(key);
        if (idx !== -1) { if (row.clusterMAs.length > 2) row.clusterMAs.splice(idx, 1); }
        else row.clusterMAs.push(key);
        var c = sfPopoverHtml(row, true);
        document.getElementById('sf-popover-content').innerHTML = c;
    };

    window.sfOpenPopover = function(id, pillEl) {
        sfActivePopoverId = id;
        var pop = document.getElementById('sf-popover');
        var row = sfRows.find(function(r){ return r.id === id; });
        if (!row) return;
        var EXTRA_TITLES = { perf: 'Performance', gap: 'Gap', range: 'Range', sector: 'Sector / Industry', indrank: 'Industry Rank' };
        document.getElementById('sf-popover-title').textContent = EXTRA_TITLES[row.type] || row.type;
        document.getElementById('sf-popover-content').innerHTML = sfPopoverHtml(row, true);
        var pop = document.getElementById('sf-popover');
        pop.style.maxWidth = (row.type === 'sector') ? '380px' : '260px';
        var rect = pillEl.getBoundingClientRect();
        pop.style.top  = (rect.bottom + 6) + 'px';
        pop.style.left = Math.min(rect.left, window.innerWidth - 290) + 'px';
        pop.style.display = 'block';
    };

    window.sfClosePopover = function() {
        document.getElementById('sf-popover').style.display = 'none';
        sfActivePopoverId = null;
        sfActivePillType = null;
        sfPillWasPreExisting = false;
        sfPopoverIsNew = false;
        sfRenderPills();
        renderScans();
    };

    document.addEventListener('click', function(e) {
        var pop = document.getElementById('sf-popover');
        if (!pop || pop.style.display === 'none') return;
        if (!pop.contains(e.target) && !e.target.closest('.sf-pill')) sfCancelPill();
    });

    window.sfPopChange = function(id, field, value) {
        var row = sfRows.find(function(r){ return r.id === id; });
        if (!row) return;
        if (field === 'type') {
            row.type     = value;
            row.dir         = (value === 'ma') ? 'above' : (value === 'gap') ? 'up' : 'gt';
            row.maCondition = 'above_price';
            row.rangeCondition = value === 'range' ? 'relative' : null;
            row.sectors    = value === 'sector' ? [] : undefined;
            row.industries = value === 'sector' ? [] : undefined;
            row.indrankMode = value === 'indrank' ? 'top' : undefined;
            row.val         = value === 'rs' ? 70 : value === 'price' ? 5 : value === 'vol' ? '500K' : value === 'adr' ? 3 : value === 'cr' ? 50 : value === 'mcap' ? '1B' : value === 'rvol' ? 1 : value === 'pattern' ? 'inside_day' : value === 'fund' ? 0 : value === 'gap' ? 1 : value === 'range' ? 50 : value === 'sector' ? null : value === 'indrank' ? 15 : null;
            row.fundMetric  = value === 'fund' ? 'eps_next_y_pct' : null;
            row.patternTf   = 'd';
            row.crTf        = 'd';
            row.maType      = 'SMA';
            row.maPeriod    = 50;
            row.maType2     = 'SMA';
            row.maPeriod1   = 5;
            row.maPeriod2   = 50;
            var c = sfPopoverHtml(row, true);
            document.getElementById('sf-popover-content').innerHTML = c;
        } else {
            if (field === 'maPeriod' || field === 'maPeriod1' || field === 'maPeriod2') value = parseInt(value);
            row[field] = value;
            // For free-text input fields don't re-render the popover — it destroys the
            // input element and kills focus after every keystroke. Just update state + pills.
            var isTextInput = (field === 'val' || field === 'val2' || field === 'clusterSpread' || field === 'wk52DistMin' || field === 'wk52DistMax');
            if (!isTextInput) {
                if (field === 'maCondition') {
                    if ((value === 'above_pct' || value === 'below_pct') && (row.val === '' || row.val == null)) row.val = 0;
                    if (value === 'between_pct') { if (row.val == null) row.val = 0; if (row.val2 == null) row.val2 = 10; }
                    if (value === 'ma_cluster') { if (!row.clusterMAs) row.clusterMAs = ['SMA50','SMA200','EMA21']; if (row.clusterSpread == null) row.clusterSpread = 1; }
                    if (value === 'slope') { if (!row.slopeDir) row.slopeDir = 'rising'; row.val = 1; }
                }
                if (field === 'slopeDir') {
                    row.val = value === 'flat' ? 0.5 : 1;
                }
                if (field === 'rangeCondition') { row.val = null; }
                if (field === 'valMetric') {
                    row.val = value === 'fwd_pe' ? 20 : value === 'ps_ratio' ? 4 : value === 'peg_ratio' ? 1 : 20;
                }
                var c2 = sfPopoverHtml(row, true);
                document.getElementById('sf-popover-content').innerHTML = c2;
            }
        }
        // Don't call sfRenderPills here — it destroys the pill bar while the popover
        // is open, which breaks the outside-click listener and resets the popover.
        // Pills are refreshed when the popover closes (sfClosePopover / sfCancelPill).
    };

    window.sfAddRow = function() {
        // Remove any existing picker
        var existing = document.getElementById('sf-add-picker');
        if (existing) { existing.remove(); return; }

        var TYPES = [
            { value: 'ma',      label: 'Moving Average' },
            { value: 'rs',      label: 'RS' },
            { value: 'perf',    label: 'Performance' },
            { value: 'price',   label: 'Price' },
            { value: 'vol',     label: 'Avg Vol' },
            { value: 'adr',     label: 'ADR%' },
            { value: 'cr',      label: 'CR%' },
            { value: 'mcap',    label: 'Market Cap' },
            { value: 'rvol',    label: 'Rel. Volume' },
            { value: 'udv',     label: 'U/D Vol' },
            { value: 'pattern', label: 'Pattern' },
            { value: 'fund',      label: 'Fundamental' },
            { value: 'valuation', label: 'Valuation' },
            { value: 'gap',     label: 'Gap' },
            { value: 'range',   label: 'Range' },
            { value: 'sector',  label: 'Sector' },
            { value: 'wk52',    label: '52-Week High/Low' },
            { value: 'indrank', label: 'Industry Rank' },
            { value: 'rsi',     label: 'RSI' },
        ];

        var picker = document.createElement('div');
        picker.id = 'sf-add-picker';
        picker.style.cssText = 'position:fixed;z-index:9999;background:#161b22;border:1px solid #30363d;border-radius:7px;min-width:180px;box-shadow:0 8px 24px rgba(0,0,0,0.5);padding:4px 0;';

        var PICKER_TOOLTIPS = {
            ma:      'Moving Average — filter by price position relative to a moving average.',
            rs:      'Relative Strength percentile vs all stocks (1–99). Higher = stronger price action.',
            perf:    'Price performance over a selected timeframe.',
            price:   'Filter by current stock price.',
            vol:     'Average daily trading volume over 50 days.',
            adr:     'Average Daily Range % — measures a stock\'s typical daily price volatility.',
            cr:      'Closing Range % — where price closed within the day\'s high/low. 100% = closed at the high.',
            mcap:    'Market Cap — total market value of the company.',
            rvol:    'Relative Volume — today\'s volume vs the average for this time of day.',
            udv:     'Up/Down Volume ratio — compares volume on up days vs down days over a selected period.',
            pattern: 'Filter by candlestick or price pattern.',
            fund:    'Fundamental metrics — EPS growth, sales growth, margins and more.',
            valuation: 'Valuation ratios — Forward P/E, P/S, and PEG.',
            gap:     'Filter stocks that gapped up or down at the open.',
            range:   'Filter by where price sits within its 52-week or recent range.',
            sector:  'Filter by market sector or specific industry.',
            wk52:    'Filter by 52-week high or low — new highs/lows and distance from each.',
            indrank: 'Filter by industry rank — show only stocks in top-ranked or lower-ranked industries.',
            rsi:     'RSI — Relative Strength Index over 14 days. Filter above or below a threshold (1–99).',
        };
        picker.innerHTML = TYPES.map(function(t) {
            var tip = PICKER_TOOLTIPS[t.value] || '';
            return '<div class="sf-add-picker-item" data-rs-tip="' + tip + '" onclick="sfAddPickerSelect(\'' + t.value + '\')" style="padding:7px 14px;font-size:0.825em;color:#c8d0dc;cursor:pointer;" onmouseover="this.style.background=\'#21262d\'" onmouseout="this.style.background=\'\'">' + t.label + '</div>';
        }).join('');

        document.body.appendChild(picker);

        // Position below the + Filter button
        var btn = document.querySelector('.sf-pill-add');
        if (btn) {
            var rect = btn.getBoundingClientRect();
            picker.style.top  = (rect.bottom + 4) + 'px';
            picker.style.left = Math.min(rect.left, window.innerWidth - 200) + 'px';
        }

        // Close on outside click
        setTimeout(function() {
            document.addEventListener('click', function closePicker(e) {
                if (!picker.contains(e.target) && !e.target.closest('.sf-pill-add')) {
                    picker.remove();
                    document.removeEventListener('click', closePicker);
                }
            });
        }, 0);
    };

    window.sfAddPickerSelect = function(type) {
        var picker = document.getElementById('sf-add-picker');
        if (picker) picker.remove();

        sfRowId++;
        var defaults = { id: sfRowId, type: type, dir: type === 'gap' ? 'up' : 'gt',
            maCondition: 'above_price', maType: 'SMA', maPeriod: 50,
            maType2: 'SMA', maPeriod1: 5, maPeriod2: 50,
            rangeCondition: type === 'range' ? 'relative' : null,
            fundMetric: type === 'fund' ? 'eps_next_y_pct' : null,
            valMetric: type === 'valuation' ? 'fwd_pe' : null,
            rsMetric: 'Percentile',
            perfTf: '1d', perfDir: 'up', patternTf: 'd', crTf: 'd', udvPeriod: 50,
            sectors: type === 'sector' ? [] : undefined,
            industries: type === 'sector' ? [] : undefined,
            wk52Side: type === 'wk52' ? 'high' : undefined,
            wk52NewOnly: type === 'wk52' ? 0 : undefined,
            wk52DistMin: type === 'wk52' ? null : undefined,
            wk52DistMax: type === 'wk52' ? null : undefined,
            indrankMode: type === 'indrank' ? 'top' : undefined,
            val: type === 'rs' ? 70 : type === 'price' ? 5 : type === 'vol' ? '500K' :
                 type === 'adr' ? 3 : type === 'cr' ? 50 : type === 'mcap' ? '1B' :
                 type === 'rvol' ? 1 : type === 'udv' ? 1.5 : type === 'pattern' ? 'inside_day' : type === 'gap' ? 1 : type === 'range' ? 50 : type === 'perf' ? 5 : type === 'sector' ? null : type === 'indrank' ? 15 : type === 'rsi' ? 50 : type === 'valuation' ? 20 : null };
        sfRows.push(defaults);
        sfRenderPills();

        // Open popover for the newly added row using the old popover system
        setTimeout(function() {
            var pills = document.querySelectorAll('.sf-pill');
            for (var i = 0; i < pills.length; i++) {
                if (pills[i].getAttribute('onclick') && pills[i].getAttribute('onclick').indexOf(sfRowId) !== -1) {
                    sfPopoverIsNew = true;
                    sfOpenPopover(sfRowId, pills[i]);
                    return;
                }
            }
        }, 30);
    };

    window.sfRemoveRow = function(id) {
        sfRows = sfRows.filter(function(r){ return r.id !== id; });
        if (sfActivePopoverId === id) sfClosePopover();
        sfRenderPills();
        renderScans();
    };

    // ── Scan selection & export ──────────────────────────────────────────
    // ── Industry stocks selection & export ───────────────────────────────
    var selectedIndustryStocks = new Set();

    var SVG_IND_PLUS  = '<svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="#484f58" stroke-width="2"><line x1="5" y1="1" x2="5" y2="9"/><line x1="1" y1="5" x2="9" y2="5"/></svg>';
    var SVG_IND_CHECK = '<svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="#3fb950" stroke-width="2.2"><polyline points="1.5,6 4.5,9 10.5,3"/></svg>';

    function indUpdateExportBtn() {
        var btn = document.getElementById('ind-export-btn');
        if (btn) btn.disabled = selectedIndustryStocks.size === 0;
    }

    window.indToggleAdd = function(btn) {
        var ticker = btn.getAttribute('data-ticker');
        if (selectedIndustryStocks.has(ticker)) {
            selectedIndustryStocks.delete(ticker);
            btn.innerHTML = SVG_IND_PLUS;
            btn.classList.remove('added');
        } else {
            selectedIndustryStocks.add(ticker);
            btn.innerHTML = SVG_IND_CHECK;
            btn.classList.add('added');
        }
        indUpdateExportBtn();
    };

    window.indToggleSelectAll = function() {
        var chk = document.getElementById('ind-select-all-chk');
        var allVisible = Array.from(document.querySelectorAll('#stocks-tbody .stock-row'))
            .filter(function(r) { return r.style.display !== 'none'; });
        var allSelected = allVisible.length > 0 && allVisible.every(function(row) {
            var btn = row.querySelector('.scan-add-btn');
            return btn && selectedIndustryStocks.has(btn.getAttribute('data-ticker'));
        });
        if (allSelected) {
            allVisible.forEach(function(row) {
                var btn = row.querySelector('.scan-add-btn');
                if (!btn) return;
                selectedIndustryStocks.delete(btn.getAttribute('data-ticker'));
                btn.classList.remove('added');
                btn.innerHTML = SVG_IND_PLUS;
            });
            if (chk) { chk.checked = false; chk.style.opacity = '0.35'; }
        } else {
            allVisible.forEach(function(row) {
                var btn = row.querySelector('.scan-add-btn');
                if (!btn) return;
                var ticker = btn.getAttribute('data-ticker');
                if (!selectedIndustryStocks.has(ticker)) {
                    selectedIndustryStocks.add(ticker);
                    btn.classList.add('added');
                    btn.innerHTML = SVG_IND_CHECK;
                }
            });
            if (chk) { chk.checked = true; chk.style.opacity = '1'; }
        }
        indUpdateExportBtn();
    };

    window.indOpenExport = function() {
        var tickers = Array.from(selectedIndustryStocks);
        document.getElementById('ind-export-sub').textContent =
            tickers.length + ' selected ticker' + (tickers.length !== 1 ? 's' : '');
        document.getElementById('ind-export-textarea').value = tickers.join(',');
        document.getElementById('ind-export-modal').classList.add('open');
    };

    window.indCloseExport = function() {
        document.getElementById('ind-export-modal').classList.remove('open');
    };

    window.indClearExport = function() {
        selectedIndustryStocks.clear();
        document.querySelectorAll('#stocks-tbody .scan-add-btn').forEach(function(btn) {
            btn.classList.remove('added');
            btn.innerHTML = SVG_IND_PLUS;
        });
        var chk = document.getElementById('ind-select-all-chk');
        if (chk) { chk.checked = false; chk.style.opacity = '0.35'; }
        indUpdateExportBtn();
        indCloseExport();
    };

    window.indCopyExport = function() {
        var ta = document.getElementById('ind-export-textarea');
        ta.select();
        document.execCommand('copy');
        var copied = document.getElementById('ind-export-copied');
        copied.classList.add('show');
        setTimeout(function() { copied.classList.remove('show'); }, 2000);
    };

    var selectedScans = new Set();

    function sfUpdateExportBtn() {
        var btn = document.getElementById('scans-export-btn');
        if (btn) btn.disabled = selectedScans.size === 0;
    }

    var SVG_PLUS  = '<svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="#484f58" stroke-width="2"><line x1="5" y1="1" x2="5" y2="9"/><line x1="1" y1="5" x2="9" y2="5"/></svg>';
    var SVG_CHECK = '<svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="#3fb950" stroke-width="2.2"><polyline points="1.5,6 4.5,9 10.5,3"/></svg>';

    window.sfToggleAdd = function(btn) {
        var ticker = btn.getAttribute('data-ticker');
        if (selectedScans.has(ticker)) {
            selectedScans.delete(ticker);
            btn.innerHTML = SVG_PLUS;
            btn.classList.remove('added');
        } else {
            selectedScans.add(ticker);
            btn.innerHTML = SVG_CHECK;
            btn.classList.add('added');
        }
        sfUpdateExportBtn();
    };

    window.sfOpenExport = function() {
        var tickers = Array.from(selectedScans);
        var total = document.querySelectorAll('#scans-tbody .stock-row').length;
        document.getElementById('sf-export-sub').textContent = tickers.length + ' selected ticker' + (tickers.length !== 1 ? 's' : '');
        document.getElementById('sf-export-textarea').value = tickers.join(',');
        document.getElementById('sf-export-modal').classList.add('open');
    };

    window.sfCloseExport = function() {
        document.getElementById('sf-export-modal').classList.remove('open');
    };

    window.sfClearExport = function() {
        selectedScans.clear();
        document.querySelectorAll('.scan-add-btn').forEach(function(btn) {
            btn.classList.remove('added');
            btn.innerHTML = SVG_PLUS;
        });
        var chk = document.getElementById('scans-select-all-chk');
        if (chk) chk.checked = false;
        sfUpdateExportBtn();
        sfCloseExport();
    };

    window.sfSelectAll = function() {
        document.querySelectorAll('#scans-tbody .stock-row').forEach(function(row) {
            if (row.style.display === 'none') return;
            var btn = row.querySelector('.scan-add-btn');
            if (!btn) return;
            var ticker = btn.getAttribute('data-ticker');
            if (!selectedScans.has(ticker)) {
                selectedScans.add(ticker);
                btn.classList.add('added');
                btn.innerHTML = SVG_CHECK;
            }
        });
        var chk = document.getElementById('scans-select-all-chk');
        if (chk) chk.checked = true;
        sfUpdateExportBtn();
    };

    window.sfToggleSelectAll = function(el) {
        // el is either the checkbox or the th — derive checked state
        var chk = document.getElementById('scans-select-all-chk');
        var allVisible = Array.from(document.querySelectorAll('#scans-tbody .stock-row'))
            .filter(function(r){ return r.style.display !== 'none'; });
        var allSelected = allVisible.every(function(row) {
            var btn = row.querySelector('.scan-add-btn');
            return btn && selectedScans.has(btn.getAttribute('data-ticker'));
        });
        if (allSelected) {
            // deselect all
            allVisible.forEach(function(row) {
                var btn = row.querySelector('.scan-add-btn');
                if (!btn) return;
                selectedScans.delete(btn.getAttribute('data-ticker'));
                btn.classList.remove('added');
                btn.innerHTML = SVG_PLUS;
            });
            if (chk) chk.checked = false;
        } else {
            sfSelectAll();
        }
        sfUpdateExportBtn();
    };

    window.sfCopyExport = function() {
        var ta = document.getElementById('sf-export-textarea');
        ta.select();
        document.execCommand('copy');
        var copied = document.getElementById('sf-export-copied');
        copied.classList.add('show');
        setTimeout(function(){ copied.classList.remove('show'); }, 2000);
    };

    // ── Presets (persistent storage) ─────────────────────────────────────
    // ── Preset helpers ───────────────────────────────────────────────────
    var LS_PRESETS_KEY = 'dashboard-scan-presets';
    var _activePresetName = null;

    function sfGetAllPresets() {
        try { return JSON.parse(localStorage.getItem(LS_PRESETS_KEY) || '{}'); }
        catch(e) { return {}; }
    }

    function sfSaveAllPresets(obj) {
        try { localStorage.setItem(LS_PRESETS_KEY, JSON.stringify(obj)); } catch(e) {}
        kvSet('scan_presets', JSON.stringify(obj));
    }

    window.sfOpenSaveModal = function() {
        if (!sfRows.length) { alert('Add at least one filter before saving.'); return; }
        document.getElementById('sf-save-input').value = '';
        document.getElementById('sf-save-modal').classList.add('open');
        setTimeout(function(){ document.getElementById('sf-save-input').focus(); }, 50);
    };

    window.sfCloseSaveModal = function() {
        document.getElementById('sf-save-modal').classList.remove('open');
    };

    window.sfSavePreset = function() {
        var name = (document.getElementById('sf-save-input').value || '').trim();
        if (!name) { alert('Please enter a preset name.'); return; }
        var presets = sfGetAllPresets();
        presets[name] = { name: name, rows: sfRows, saved: Date.now() };
        sfSaveAllPresets(presets);
        sfCloseSaveModal();
    };

    window.sfTogglePresets = function(btn) {
        var dd = document.getElementById('sf-preset-dropdown');
        if (dd.classList.contains('open')) { dd.classList.remove('open'); return; }
        var rect = btn.getBoundingClientRect();
        var vpH  = window.innerHeight;
        var vpW  = window.innerWidth;
        var ddW  = 280;
        var ddMaxH = 320;
        sfRenderPresetsDropdown();
        // Anchor right edge of dropdown to right edge of button, never off-screen
        var btnRight = rect.right;
        var dropRight = vpW - btnRight; // distance from viewport right to button right
        dd.style.right = Math.max(8, vpW - btnRight) + 'px';
        dd.style.left  = 'auto';
        var spaceBelow = vpH - rect.bottom - 8;
        if (spaceBelow < ddMaxH && rect.top > spaceBelow) {
            var maxH = Math.min(ddMaxH, rect.top - 8);
            dd.style.top    = 'auto';
            dd.style.bottom = (vpH - rect.top + 6) + 'px';
            dd.style.maxHeight = maxH + 'px';
        } else {
            dd.style.bottom    = 'auto';
            dd.style.top       = (rect.bottom + 6) + 'px';
            dd.style.maxHeight = Math.min(ddMaxH, spaceBelow) + 'px';
        }
        dd.classList.add('open');
    };

    function sfRenderPresetsDropdown() {
        var dd = document.getElementById('sf-preset-dropdown');
        var presets = sfGetAllPresets();
        var names = Object.keys(presets).sort();
        var scrollHtml = '<div class="sf-preset-dropdown-scroll">';
        scrollHtml += '<div style="padding:8px 12px 4px;border-bottom:1px solid #21262d;display:flex;gap:6px;">' +
            '<button class="sf-preset-btn save" style="flex:1;font-size:0.792em;padding:3px 0;" data-sf-action="export-all">Export all</button>' +
            '<label class="sf-preset-btn" style="flex:1;font-size:0.792em;padding:3px 0;text-align:center;cursor:pointer;">Import file<input type="file" accept=".json" style="display:none" onchange="sfImportFile(this)"></label>' +
            '</div>';
        var html = scrollHtml;
        if (!names.length) {
            html += '<div class="sf-preset-empty">No saved presets yet.</div>';
        } else {
            names.forEach(function(name) {
                var n = esc(name);
                var isActive = (_activePresetName === name) ? ' sf-preset-active' : '';
                html += '<div class="sf-preset-item' + isActive + '">' +
                    '<span class="sf-preset-item-name" data-sf-apply="' + n + '">' + n + '</span>' +
                    '<button class="sf-preset-delete" data-sf-export="' + n + '" title="Export" style="font-size:0.75em;color:#6e7681;padding:0 5px;">↓</button>' +
                    '<button class="sf-preset-delete" data-sf-delete="' + n + '" title="Delete">×</button>' +
                    '</div>';
            });
        }
        html += '</div>'; // close sf-preset-dropdown-scroll
        var footer = '<div class="sf-preset-dropdown-footer">' +
            '<div class="sf-preset-item" data-sf-action="copy-results" style="color:#58a6ff;cursor:pointer;flex:1;border-bottom:none;">' +
            '<span style="flex:1;">Copy results</span>' +
            '<span style="font-size:10px;color:#484f58;" id="sf-copy-feedback"></span>' +
            '</div>' +
            '<div class="sf-preset-item" data-sf-action="reset-filters" style="color:#f85149;cursor:pointer;border-left:1px solid #21262d;padding:0 12px;border-bottom:none;">' +
            'Reset' +
            '</div></div>';
        dd.innerHTML = html + footer;

        // Delegated click handler — re-attached each render
        dd.onclick = function(e) {
            e.stopPropagation();
            var applyEl  = e.target.closest('[data-sf-apply]');
            var exportEl = e.target.closest('[data-sf-export]');
            var deleteEl = e.target.closest('[data-sf-delete]');
            var exportAll    = e.target.closest('[data-sf-action="export-all"]');
            var copyResults  = e.target.closest('[data-sf-action="copy-results"]');
            var resetFilters = e.target.closest('[data-sf-action="reset-filters"]');
            if (applyEl)      { sfApplyPreset(applyEl.getAttribute('data-sf-apply')); return; }
            if (exportEl)     { sfExportOne(exportEl.getAttribute('data-sf-export')); return; }
            if (deleteEl)     { sfDeletePreset(deleteEl.getAttribute('data-sf-delete')); return; }
            if (exportAll)    { sfExportAll(); return; }
            if (copyResults)  { sfCopyResults(); return; }
            if (resetFilters) { sfResetFilters(); return; }
        };
    }

    window.sfCopyResults = function() {
        var rows = document.querySelectorAll('#scans-tbody .stock-row');
        if (!rows.length) { return; }
        var tickers = [];
        rows.forEach(function(r) {
            var sym = r.getAttribute('data-symbol');
            if (sym) tickers.push(sym);
        });
        var text = tickers.join(',');
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        var fb = document.getElementById('sf-copy-feedback');
        if (fb) { fb.textContent = 'Copied ' + tickers.length + '!'; setTimeout(function(){ fb.textContent = ''; }, 2000); }
    };

    window.sfResetFilters = function() {
        sfRows = [];
        sfRowId = 0;
        _activePresetName = null;
        var dd = document.getElementById('sf-preset-dropdown');
        if (dd) dd.classList.remove('open');
        sfRenderPills();
        renderScans();
    };

    window.sfApplyPreset = function(name) {
        var dd = document.getElementById('sf-preset-dropdown');
        if (dd) dd.classList.remove('open');
        var presets = sfGetAllPresets();
        var preset = presets[name];
        if (!preset) return;
        // Close any open popover and reset all pill state before applying
        var pop = document.getElementById('sf-popover');
        if (pop) pop.style.display = 'none';
        sfActivePillType = null;
        sfActivePopoverId = null;
        sfPillWasPreExisting = false;
        sfPopoverIsNew = false;
        _activePresetName = name;
        sfRows = preset.rows || [];
        sfRowId = sfRows.reduce(function(m, r){ return Math.max(m, r.id||0); }, 0);
        sfRenderPills();
        renderScans();
    };

    window.sfDeletePreset = function(name) {
        var presets = sfGetAllPresets();
        delete presets[name];
        sfSaveAllPresets(presets);
        sfRenderPresetsDropdown();
    };

    window.sfExportOne = function(name) {
        var presets = sfGetAllPresets();
        var preset = presets[name];
        if (!preset) return;
        sfDownloadJson({ presets: { [name]: preset } }, 'preset-' + name.replace(/[^a-z0-9]/gi, '_') + '.json');
    };

    window.sfExportAll = function() {
        var presets = sfGetAllPresets();
        sfDownloadJson({ presets: presets }, 'scan-presets-all.json');
        document.getElementById('sf-preset-dropdown').classList.remove('open');
    };

    function sfDownloadJson(obj, filename) {
        var blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    window.sfImportFile = function(input) {
        var file = input.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function(e) {
            try {
                var data = JSON.parse(e.target.result);
                if (!data.presets) { alert('Invalid preset file.'); return; }
                var existing = sfGetAllPresets();
                var imported = 0;
                Object.keys(data.presets).forEach(function(name) {
                    existing[name] = data.presets[name];
                    imported++;
                });
                sfSaveAllPresets(existing);
                document.getElementById('sf-preset-dropdown').classList.remove('open');
                alert('Imported ' + imported + ' preset' + (imported !== 1 ? 's' : '') + ' successfully.');
            } catch(err) {
                alert('Failed to read preset file.');
            }
        };
        reader.readAsText(file);
        input.value = '';
    };

    // Close preset dropdown on outside click
    document.addEventListener('click', function(e) {
        var dd = document.getElementById('sf-preset-dropdown');
        var btn = document.getElementById('sf-load-btn');
        if (dd && dd.classList.contains('open') && !dd.contains(e.target) && e.target !== btn) {
            dd.classList.remove('open');
        }
    });

    // Save modal keyboard shortcut
    document.addEventListener('keydown', function(e) {
        var modal = document.getElementById('sf-save-modal');
        if (modal && modal.classList.contains('open')) {
            if (e.key === 'Enter') sfSavePreset();
            if (e.key === 'Escape') sfCloseSaveModal();
        }
    });


    function applyFilters(stocks) {
        return stocks.filter(function(r) {
            for (var i = 0; i < sfRows.length; i++) {
                var f = sfRows[i];
                if (f.type === 'ma') {
                    var cond = f.maCondition || 'above_price';
                    var key  = (f.maType||'SMA') + (f.maPeriod||50);
                    var k1   = (f.maType||'SMA')  + (f.maPeriod1||5);
                    var k2   = (f.maType2||'SMA') + (f.maPeriod2||50);
                    if (cond === 'above_price') {
                        var dist = r.dist_ma ? r.dist_ma[key] : null;
                        if (dist == null || dist <= 0) return false;
                    } else if (cond === 'below_price') {
                        var dist = r.dist_ma ? r.dist_ma[key] : null;
                        if (dist == null || dist >= 0) return false;
                    } else if (cond === 'price_crossed_above') {
                        var xKey = key + '|above';
                        if ((r.price_ma_crossovers || []).indexOf(xKey) === -1) return false;
                    } else if (cond === 'price_crossed_below') {
                        var xKey = key + '|below';
                        if ((r.price_ma_crossovers || []).indexOf(xKey) === -1) return false;
                    } else if (cond === 'above_pct') {
                        var dist = r.dist_ma ? r.dist_ma[key] : null;
                        var thresh = parseFloat(f.val);
                        if (dist == null || (!isNaN(thresh) && dist <= thresh)) return false;
                    } else if (cond === 'below_pct') {
                        var dist = r.dist_ma ? r.dist_ma[key] : null;
                        var thresh = parseFloat(f.val);
                        if (dist == null || (!isNaN(thresh) && dist >= thresh)) return false;
                    } else if (cond === 'between_pct') {
                        var dist = r.dist_ma ? r.dist_ma[key] : null;
                        var tMin = parseFloat(f.val);
                        var tMax = parseFloat(f.val2);
                        if (dist == null) return false;
                        if (!isNaN(tMin) && dist < tMin) return false;
                        if (!isNaN(tMax) && dist > tMax) return false;
                    } else if (cond === 'ma_above') {
                        var v1 = r.ma_val ? r.ma_val[k1] : null;
                        var v2 = r.ma_val ? r.ma_val[k2] : null;
                        if (v1 == null || v2 == null || v1 <= v2) return false;
                    } else if (cond === 'ma_below') {
                        var v1 = r.ma_val ? r.ma_val[k1] : null;
                        var v2 = r.ma_val ? r.ma_val[k2] : null;
                        if (v1 == null || v2 == null || v1 >= v2) return false;
                    } else if (cond === 'crosses_above') {
                        var xKey = k1 + '|' + k2 + '|above';
                        if ((r.ma_crossovers || []).indexOf(xKey) === -1) return false;
                    } else if (cond === 'crosses_below') {
                        var xKey = k1 + '|' + k2 + '|below';
                        if ((r.ma_crossovers || []).indexOf(xKey) === -1) return false;
                    } else if (cond === 'ma_cluster') {
                        var cms   = f.clusterMAs || ['SMA50','SMA200','EMA21'];
                        var spread = parseFloat(f.clusterSpread != null ? f.clusterSpread : 1);
                        if (isNaN(spread)) spread = 1;
                        var vals = cms.map(function(k){ return r.ma_val ? r.ma_val[k] : null; });
                        if (vals.some(function(v){ return v == null; })) return false;
                        var minV = Math.min.apply(null, vals);
                        var maxV = Math.max.apply(null, vals);
                        if (minV <= 0) return false;
                        if (((maxV - minV) / minV * 100) > spread) return false;
                    } else if (cond === 'slope') {
                        var slopeDir = f.slopeDir || 'rising';
                        var slope = r.slope_ma ? r.slope_ma[key] : null;
                        var thresh = parseFloat(f.val);
                        if (slope == null) return false;
                        if (slopeDir === 'rising'  && !isNaN(thresh) && slope < thresh)        return false;
                        if (slopeDir === 'falling' && !isNaN(thresh) && slope > -thresh)       return false;
                        if (slopeDir === 'flat'    && !isNaN(thresh) && Math.abs(slope) > thresh) return false;
                    }
                } else if (f.type === 'rs') {
                    var rsField = f.rsMetric || 'Percentile';
                    var rs = r[rsField];
                    if (rs == null) return false;
                    var rv = parseFloat(f.val);
                    if (!isNaN(rv)) {
                        if (f.dir === 'gt' && rs <= rv) return false;
                        if (f.dir === 'lt' && rs >= rv) return false;
                    }
                } else if (f.type === 'price') {
                    var pr = r.price;
                    if (pr == null) return false;
                    var pv = parseFloat(f.val);
                    if (!isNaN(pv)) {
                        if (f.dir === 'gt' && pr <= pv) return false;
                        if (f.dir === 'lt' && pr >= pv) return false;
                    }
                } else if (f.type === 'vol') {
                    var vol = r.AvgVol50;
                    if (vol == null) return false;
                    var vv = sfParseVol(f.val);
                    if (!isNaN(vv)) {
                        if (f.dir === 'gt' && vol <= vv) return false;
                        if (f.dir === 'lt' && vol >= vv) return false;
                    }
                } else if (f.type === 'adr') {
                    var adr = r.adr_pct;
                    if (adr == null) return false;
                    var av = parseFloat(f.val);
                    if (!isNaN(av)) {
                        if (f.dir === 'gt' && adr <= av) return false;
                        if (f.dir === 'lt' && adr >= av) return false;
                    }
                } else if (f.type === 'pattern') {
                    var ptfSuffix = (f.patternTf === 'w') ? '_w' : (f.patternTf === 'm') ? '_m' : '';
                    var patKey = f.val + ptfSuffix;
                    if (!r[patKey]) return false;
                } else if (f.type === 'rvol') {
                    var rv = r.rel_vol;
                    if (rv == null) return false;
                    var rvThresh = parseFloat(f.val);
                    if (!isNaN(rvThresh)) {
                        if (f.dir === 'gt' && rv <= rvThresh) return false;
                        if (f.dir === 'lt' && rv >= rvThresh) return false;
                    }
                } else if (f.type === 'udv') {
                    var udvField = (f.udvPeriod === 20) ? 'ud_vol_ratio_20' : 'ud_vol_ratio_50';
                    var udv = r[udvField];
                    if (udv == null) return false;
                    var udvThresh = parseFloat(f.val);
                    if (!isNaN(udvThresh) && udv <= udvThresh) return false;
                } else if (f.type === 'cr') {
                    var crField = (f.crTf === 'w') ? 'cr_w' : (f.crTf === 'm') ? 'cr_m' : 'cr';
                    var cr;
                    if (!f.crTf || f.crTf === 'd') {
                        var _liveCrF = scanLivePrices[r.ticker];
                        cr = (_liveCrF && _liveCrF.price && _liveCrF.dayHigh != null && _liveCrF.dayLow != null && _liveCrF.dayHigh > _liveCrF.dayLow)
                            ? ((_liveCrF.price - _liveCrF.dayLow) / (_liveCrF.dayHigh - _liveCrF.dayLow)) * 100
                            : r[crField];
                    } else {
                        cr = r[crField];
                    }
                    if (cr == null) return false;
                    var cv = parseFloat(f.val);
                    if (!isNaN(cv)) {
                        if (f.dir === 'gt' && cr <= cv) return false;
                        if (f.dir === 'lt' && cr >= cv) return false;
                    }
                } else if (f.type === 'mcap') {
                    var mc = r.MarketCap;
                    if (mc == null) return false;
                    var mv = sfParseMcap(f.val);
                    if (!isNaN(mv)) {
                        if (f.dir === 'gt' && mc <= mv) return false;
                        if (f.dir === 'lt' && mc >= mv) return false;
                    }
                } else if (f.type === 'fund') {
                    var metric = f.fundMetric || 'eps_next_y_pct';
                    var fv = r[metric];
                    if (fv == null) return false;
                    var fthresh = parseFloat(f.val);
                    if (!isNaN(fthresh)) {
                        if (f.dir === 'gt' && fv <= fthresh) return false;
                        if (f.dir === 'lt' && fv >= fthresh) return false;
                    }
                } else if (f.type === 'valuation') {
                    var valMetric = f.valMetric || 'fwd_pe';
                    var vv = r[valMetric];
                    if (vv == null || vv <= 0) return false;
                    var vthresh = parseFloat(f.val);
                    if (!isNaN(vthresh)) {
                        if (f.dir === 'gt' && vv < vthresh) return false;
                        if (f.dir === 'lt' && vv > vthresh) return false;
                    }
                } else if (f.type === 'gap') {
                    var gv = r.gap_pct;
                    if (gv == null) return false;
                    var gthresh = parseFloat(f.val);
                    if (isNaN(gthresh)) gthresh = 0;
                    if (f.dir === 'up'   && gv < gthresh)  return false;
                    if (f.dir === 'down' && gv > -gthresh) return false;
                } else if (f.type === 'perf') {
                    var perfTf = f.perfTf || '1d';
                    var perfField = perfTf === '1d' ? 'daily' : perfTf === '1w' ? '1w' : perfTf === '1m' ? '1m' : '3m';
                    var pv;
                    if (perfTf === '1d') {
                        var _livePerf = scanLivePrices[r.ticker];
                        pv = (wlIsMarketOpen() && _livePerf && _livePerf.price && _livePerf.prevClose)
                            ? ((_livePerf.price - _livePerf.prevClose) / _livePerf.prevClose) * 100
                            : r[perfField];
                    } else {
                        pv = r[perfField];
                    }
                    if (pv == null) return false;
                    var pthresh = parseFloat(f.val);
                    if (isNaN(pthresh)) pthresh = 0;
                    if (f.perfDir === 'up'   && pv < pthresh)  return false;
                    if (f.perfDir === 'down' && pv > -pthresh) return false;
                } else if (f.type === 'range') {
                    var rc = f.rangeCondition || 'relative';
                    if (f.val == null || f.val === '') return true;
                    var rval = parseFloat(f.val);
                    if (isNaN(rval)) return true;
                    if (rc === 'relative') {
                        var rva = r.range_vs_adr;
                        if (rva == null) return false;
                        if (rva > rval) return false;
                    } else {
                        var rrk = r.range_rank;
                        if (rrk == null) return false;
                        if (rrk < rval) return false;
                    }
                } else if (f.type === 'sector') {
                    var secs = f.sectors || [];
                    var inds = f.industries || [];
                    if (inds.length > 0) {
                        if (inds.indexOf(r.industry) === -1) return false;
                    } else if (secs.length > 0) {
                        if (secs.indexOf(r.sector) === -1) return false;
                    }
                } else if (f.type === 'wk52') {
                    var wSide    = f.wk52Side || 'high';
                    var wNewOnly = !!f.wk52NewOnly;
                    var wDistMin = f.wk52DistMin != null && f.wk52DistMin !== '' ? parseFloat(f.wk52DistMin) : null;
                    var wDistMax = f.wk52DistMax != null && f.wk52DistMax !== '' ? parseFloat(f.wk52DistMax) : null;
                    if (wSide === 'high') {
                        if (wNewOnly && !r.new_52wk_high) return false;
                        if (!wNewOnly && (wDistMin !== null || wDistMax !== null)) {
                            var pct = r.PctFrom52WkHigh != null ? r.PctFrom52WkHigh : null;
                            if (pct == null || pct >= 0) return false;
                            if (wDistMin !== null && !isNaN(wDistMin) && pct > -wDistMin) return false;
                            if (wDistMax !== null && !isNaN(wDistMax) && pct < -wDistMax) return false;
                        }
                    } else {
                        if (wNewOnly && !r.new_52wk_low) return false;
                        if (!wNewOnly && (wDistMin !== null || wDistMax !== null)) {
                            var distLow = r.PctFrom52WkLow != null ? r.PctFrom52WkLow : null;
                            if (distLow == null) return false;
                            if (wDistMin !== null && !isNaN(wDistMin) && distLow < wDistMin) return false;
                            if (wDistMax !== null && !isNaN(wDistMax) && distLow > wDistMax) return false;
                        }
                    }
                } else if (f.type === 'indrank') {
                    var irRank = _indRankMap[r.industry];
                    if (!irRank || irRank === 9999) return false;
                    var irVal = parseInt(f.val);
                    if (!isNaN(irVal)) {
                        if ((f.indrankMode || 'top') === 'top'   && irRank > irVal)  return false;
                        if ((f.indrankMode || 'top') === 'below' && irRank <= irVal) return false;
                    }
                } else if (f.type === 'rsi') {
                    var rsiVal = r.rsi14;
                    if (rsiVal == null) return false;
                    var rsiThresh = parseFloat(f.val);
                    if (!isNaN(rsiThresh)) {
                        if (f.dir === 'gt' && rsiVal <= rsiThresh) return false;
                        if (f.dir === 'lt' && rsiVal >= rsiThresh) return false;
                    }
                }
            }
            return true;
        });
    }

    window.setScan = function(scan) {
        activeScan = (activeScan === scan) ? null : scan;
        document.querySelectorAll('.scan-filter-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-scan') === activeScan);
        });
        renderScans();
    };

    function buildScansHeader() {
        var dmaTooltip = '% distance from ' + activeMAType + activeMALength;
        var cols = [
            { key:'symbol',    label:'Ticker',  tip:'Ticker symbol' },
            { key:'industry',  label:'Industry',tip:'Industry (sort by rank)' },
            { key:'price',     label:'Price',   tip:'Last closing price' },
            { key:'rs',        label:'RS',      tip:'RS Percentile (1-99)' },
            { key:'weighted_rs_pct', label:'3M RS', tip:'Weighted 3M RS Percentile' },
            { key:'chg',       label:'Chg',     tip:'Daily change ($)' },
            { key:'daily',     label:'Chg%',    tip:'Daily return %' },
            { key:'1w',        label:'1W',      tip:'5-day return' },
            { key:'1m',        label:'1M',      tip:'21-day return' },
            { key:'3m',        label:'3M',      tip:'63-day return' },
            { key:'ytd',       label:'1Y',      tip:'1-year return' },
            { key:'vs_spy',    label:'vs 1M',   tip:'1M return vs SPX' },
            { key:'vs_spy_3m', label:'vs 3M',   tip:'3M return vs SPX' },
            { key:'dist_ma',   label:'Dist/MA', tip:dmaTooltip, extra:'<span class="dist-ma-btn" onclick="event.stopPropagation();toggleDistMA(this)">⋯</span>' },
            { key:'avg_vol',   label:'Avg Vol', tip:'50-day average volume' },
            { key:'pct_52wk',  label:'52Wk%',   tip:'% from 52-week high' },
            { key:'adr_pct',   label:'ADR%',    tip:'Avg Daily Range %' },
            { key:'cr',        label:'CR',      tip:'Closing range (100=high, 0=low)' },
        ];
        var fundCols = [
            { key:'eps_this_y_pct',    label:'EPS TY',    tip:'EPS Growth This Year %' },
            { key:'eps_next_y_pct',    label:'EPS NY',    tip:'EPS Growth Next Year %' },
            { key:'eps_next_5y_pct',   label:'EPS 5Y',    tip:'EPS Growth Next 5 Years %' },
            { key:'eps_qoq_pct',       label:'EPS Q/Q',   tip:'EPS Growth Qtr over Qtr %' },
            { key:'sales_qoq_pct',     label:'Sales Q/Q', tip:'Sales Growth Qtr over Qtr %' },
            { key:'profit_margin_pct', label:'Margin',    tip:'Profit Margin %' },
        ];
        var html = '<tr>';
        html += '<th style="text-align:left;padding-left:8px;cursor:pointer;" title="Select / deselect all" onclick="sfToggleSelectAll(this)">' +
            '<input type="checkbox" id="scans-select-all-chk" onclick="event.stopPropagation();sfToggleSelectAll(document.getElementById(\'scans-select-all-chk\'))" style="cursor:pointer;accent-color:#388bfd;color-scheme:dark;opacity:0.35;">' +
            '</th>';
        cols.slice(1).forEach(function(c) {
            var sorted = scansSortState.by === c.key;
            var cl = sorted ? (' sorted ' + (scansSortState.dir === -1 ? 'sort-desc' : 'sort-asc')) : '';
            html += '<th class="sortable' + cl + '" data-sort-by="' + c.key + '" data-tooltip="' + esc(c.tip) + '">' + c.label + (c.extra||'') + '</th>';
        });
        fundCols.forEach(function(c) {
            var sorted = scansSortState.by === c.key;
            var cl = sorted ? (' sorted ' + (scansSortState.dir === -1 ? 'sort-desc' : 'sort-asc')) : '';
            html += '<th class="sortable' + cl + '" data-sort-by="' + c.key + '" data-tooltip="' + esc(c.tip) + '">' + c.label + '</th>';
        });
        html += '</tr>';
        document.getElementById('scans-thead').innerHTML = html;

        document.getElementById('scans-thead').querySelectorAll('th[data-sort-by]').forEach(function(th) {
            th.addEventListener('click', function() {
                var key = this.getAttribute('data-sort-by');
                if (scansSortState.by === key) { scansSortState.dir *= -1; }
                else { scansSortState.by = key; scansSortState.dir = key === 'industry' ? 1 : -1; }
                renderScans();
            });
        });
    }

    var _renderScansTimer = null;

    // ── Virtual scroll state ──────────────────────────────────────────────
    var _vsData          = [];   // full sorted+filtered row data
    var _vsRowHeight     = 34;   // px per row (measured after first render)
    var _vsBuffer        = 15;   // extra rows above/below viewport
    var _vsRenderedStart = -1;
    var _vsRenderedEnd   = -1;
    var _vsScrollBound   = false;

    function renderScans() {
        if (_renderScansTimer) clearTimeout(_renderScansTimer);
        _renderScansTimer = setTimeout(_doRenderScans, 60);
    }

    function _buildScanRow(row) {
        var rsVal    = row.Percentile != null ? row.Percentile : '\u2014';
        var priceVal = row.price != null ? '$' + row.price.toFixed(2) : '\u2014';
        var distAll  = row.dist_ma || {};
        var adrVal   = row.adr_pct != null ? row.adr_pct.toFixed(1)+'%' : '\u2014';
        var volVal   = '\u2014';
        if (row.AvgVol50 != null) {
            var v = row.AvgVol50;
            volVal = v >= 1e6 ? (v/1e6).toFixed(1)+'M' : v >= 1e3 ? (v/1e3).toFixed(0)+'K' : v.toFixed(0);
        }
        var adrColor = '#484f58';
        if (row.adr_pct != null) {
            if (row.adr_pct < 4) adrColor = '#3fb950';
            else if (row.adr_pct < 8) adrColor = '#e3852b';
            else adrColor = '#f85149';
        }
        var _scanLiveForCr = scanLivePrices[row.ticker];
        var _liveCrVal = null;
        if (_scanLiveForCr && _scanLiveForCr.price && _scanLiveForCr.dayHigh != null && _scanLiveForCr.dayLow != null && _scanLiveForCr.dayHigh > _scanLiveForCr.dayLow) {
            _liveCrVal = ((_scanLiveForCr.price - _scanLiveForCr.dayLow) / (_scanLiveForCr.dayHigh - _scanLiveForCr.dayLow)) * 100;
        }
        var _crDisplay = _liveCrVal != null ? _liveCrVal : row.cr;
        var crVal   = _crDisplay != null ? Math.round(_crDisplay)+'%' : '\u2014';
        var isAdded = selectedScans.has(row.ticker);
        var addIcon = isAdded
            ? '<svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="#3fb950" stroke-width="2.2"><polyline points="1.5,6 4.5,9 10.5,3"/></svg>'
            : '<svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="#484f58" stroke-width="2"><line x1="5" y1="1" x2="5" y2="9"/><line x1="1" y1="5" x2="9" y2="5"/></svg>';
        var wrsVal2 = row.weighted_rs_pct != null ? Math.round(row.weighted_rs_pct) : '\u2014';
        var _scanLive = scanLivePrices[row.ticker];
        var _usePrice = (_scanLive && _scanLive.price) ? _scanLive.price : row.price;
        var chgAbs, _dailyPct;
        if (wlIsMarketOpen() && _scanLive && _scanLive.price && _scanLive.prevClose) {
            chgAbs    = _scanLive.price - _scanLive.prevClose;
            _dailyPct = (chgAbs / _scanLive.prevClose) * 100;
        } else {
            chgAbs    = (row.price != null && row.daily != null) ? (row.price / (1 + row.daily / 100)) * (row.daily / 100) : null;
            _dailyPct = row.daily;
        }
        var chgStr  = chgAbs != null ? (chgAbs >= 0 ? '+' : '') + chgAbs.toFixed(2) : '\u2014';
        // ── Live Dist/MA inline ──────────────────────────────────────────
        var _dmaKey      = activeMAType + activeMALength;
        var _snapForDist = tickerMap && tickerMap[row.ticker];
        var _snapPriceD  = _snapForDist ? (_snapForDist._snapPrice != null ? _snapForDist._snapPrice : _snapForDist.price) : null;
        var _snapDistVal = (row.dist_ma && row.dist_ma[_dmaKey] != null) ? row.dist_ma[_dmaKey] : null;
        var _distDisplay;
        if (_usePrice && _snapPriceD && _snapDistVal != null) {
            var _maVal   = _snapPriceD / (1 + _snapDistVal / 100);
            _distDisplay = (_usePrice - _maVal) / _maVal * 100;
        } else {
            _distDisplay = _snapDistVal;
        }
        var _distHtml = _distDisplay != null
            ? '<span class="' + (_distDisplay > 0 ? 'up' : _distDisplay < 0 ? 'down' : '') + '">' + fmt(_distDisplay, 2, '%') + '</span>'
            : '<span style="color:#30363d">\u2014</span>';
        var pct52Val = '\u2014', pct52Color = '#484f58';
        if (row.PctFrom52WkHigh != null) {
            pct52Val   = (row.PctFrom52WkHigh > 0 ? '+' : '') + row.PctFrom52WkHigh.toFixed(1) + '%';
            pct52Color = row.PctFrom52WkHigh >= -5 ? '#3fb950' : row.PctFrom52WkHigh >= -15 ? '#e3852b' : '#f85149';
        }
        var crColor = '#484f58';
        if (_crDisplay != null) { if (_crDisplay >= 60) crColor = '#3fb950'; else if (_crDisplay >= 30) crColor = '#e3852b'; else crColor = '#f85149'; }

        var h = '<tr class="stock-row" data-symbol="' + esc(row.ticker) + '" data-sector="' + esc(row.sector||'') + '" data-industry="' + esc(row.industry||'') + '">';
        h += '<td onclick="event.stopPropagation()" style="white-space:nowrap;">' +
            '<button class="scan-add-btn' + (isAdded ? ' added' : '') + '" data-ticker="' + esc(row.ticker) + '" onclick="event.stopPropagation();sfToggleAdd(this)" title="Select for export">' + addIcon + '</button>' +
            '<button class="wl-add-btn" data-ticker="' + esc(row.ticker) + '" onclick="event.stopPropagation();wlQuickToggle(this)" title="Add to watchlist">\u2606</button>' +
            '<button class="wl-pick-btn" data-ticker="' + esc(row.ticker) + '" onclick="event.stopPropagation();wlOpenPicker(this,event)" title="Choose watchlist">\u25be</button>' +
            '<span class="ticker-badge">' + esc(row.ticker) + '</span></td>';
        var _indRk = _indRankMap[row.industry] && _indRankMap[row.industry] !== 9999 ? _indRankMap[row.industry] : null;
        var _indRkHtml = _indRk != null ? '<span style="font-size:0.858em;color:rgba(56,139,253,0.75);font-weight:600;flex-shrink:0;font-variant-numeric:tabular-nums;">#' + _indRk + '</span>' : '';
        var _indName = row.industry || '';
        var _indSpan = _indName
            ? '<span class="scan-ind-link" data-ind="' + esc(_indName) + '" onclick="event.stopPropagation();openIndustry(this.dataset.ind);" style="font-size:0.858em;color:#8b949e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;flex-shrink:1;">' + esc(_indName) + '</span>'
            : '<span style="font-size:0.858em;color:#8b949e;">\u2014</span>';
        h += '<td style="white-space:nowrap;overflow:hidden;"><div style="display:flex;align-items:baseline;gap:5px;overflow:hidden;">' + _indSpan + _indRkHtml + '</div></td>';
        h += '<td style="color:#c8d0dc;font-weight:500;">'  + (_usePrice != null ? '$' + _usePrice.toFixed(2) : '\u2014') + '</td>';
        h += '<td style="color:#c8d0dc;font-weight:600;">'  + rsVal    + '</td>';
        h += '<td style="color:#c8d0dc;font-weight:600;">'  + wrsVal2  + '</td>';
        h += '<td class="' + cc(_dailyPct)     + '">' + chgStr + '</td>';
        h += '<td class="' + cc(_dailyPct)     + '">' + (_dailyPct     != null ? fmt(_dailyPct,2,'%')     : '\u2014') + '</td>';
        h += '<td class="' + cc(row['1w'])     + '">' + (row['1w']     != null ? fmt(row['1w'],2,'%')     : '\u2014') + '</td>';
        h += '<td class="' + cc(row['1m'])     + '">' + (row['1m']     != null ? fmt(row['1m'],2,'%')     : '\u2014') + '</td>';
        h += '<td class="' + cc(row['3m'])     + '">' + (row['3m']     != null ? fmt(row['3m'],2,'%')     : '\u2014') + '</td>';
        h += '<td class="' + cc(row['1y'])     + '">' + (row['1y']     != null ? fmt(row['1y'],2,'%')     : '\u2014') + '</td>';
        h += '<td class="' + cc(row.vs_spy)    + '">' + (row.vs_spy    != null ? fmt(row.vs_spy,2,'%')    : '\u2014') + '</td>';
        h += '<td class="' + cc(row.vs_spy_3m) + '">' + (row.vs_spy_3m != null ? fmt(row.vs_spy_3m,2,'%') : '\u2014') + '</td>';
        h += '<td class="dist-ma-cell" data-dist-all="' + esc(JSON.stringify(distAll)) + '">' + _distHtml + '</td>';
        h += '<td style="color:#8b949e;">' + volVal + '</td>';
        h += '<td><span style="color:' + pct52Color + ';font-weight:600;">' + pct52Val + '</span></td>';
        h += '<td><span style="color:' + adrColor   + ';font-weight:600;">' + adrVal   + '</span></td>';
        h += '<td><span style="color:' + crColor    + ';font-weight:600;">' + crVal    + '</span></td>';
        h += '<td class="' + cc(row.eps_this_y_pct)    + '">' + (row.eps_this_y_pct    != null ? fmt(row.eps_this_y_pct,2,'%')    : '\u2014') + '</td>';
        h += '<td class="' + cc(row.eps_next_y_pct)    + '">' + (row.eps_next_y_pct    != null ? fmt(row.eps_next_y_pct,2,'%')    : '\u2014') + '</td>';
        h += '<td class="' + cc(row.eps_next_5y_pct)   + '">' + (row.eps_next_5y_pct   != null ? fmt(row.eps_next_5y_pct,2,'%')   : '\u2014') + '</td>';
        h += '<td class="' + cc(row.eps_qoq_pct)       + '">' + (row.eps_qoq_pct       != null ? fmt(row.eps_qoq_pct,2,'%')       : '\u2014') + '</td>';
        h += '<td class="' + cc(row.sales_qoq_pct)     + '">' + (row.sales_qoq_pct     != null ? fmt(row.sales_qoq_pct,2,'%')     : '\u2014') + '</td>';
        h += '<td class="' + cc(row.profit_margin_pct) + '">' + (row.profit_margin_pct != null ? fmt(row.profit_margin_pct,2,'%') : '\u2014') + '</td>';
        h += '</tr>';
        return h;
    }

    function _vsRenderWindow(scrollTop) {
        var tbody    = document.getElementById('scans-tbody');
        var data     = _vsData;
        var total    = data.length;
        if (!total) return;

        var rh       = _vsRowHeight;
        var wrap     = document.querySelector('#scans-table-view .stocks-table-wrap');
        var viewH    = wrap ? wrap.clientHeight : 600;
        var visible  = Math.ceil(viewH / rh);
        var startIdx = Math.max(0, Math.floor(scrollTop / rh) - _vsBuffer);
        var endIdx   = Math.min(total - 1, startIdx + visible + _vsBuffer * 2);

        if (startIdx === _vsRenderedStart && endIdx === _vsRenderedEnd) return;
        _vsRenderedStart = startIdx;
        _vsRenderedEnd   = endIdx;

        var topPad    = startIdx * rh;
        var bottomPad = Math.max(0, (total - endIdx - 1) * rh);

        var html = '';
        if (topPad > 0)    html += '<tr id="vs-top-spacer" style="height:' + topPad    + 'px;"><td colspan="24"></td></tr>';
        for (var i = startIdx; i <= endIdx; i++) html += _buildScanRow(data[i]);
        if (bottomPad > 0) html += '<tr id="vs-bot-spacer" style="height:' + bottomPad + 'px;"><td colspan="24"></td></tr>';

        tbody.innerHTML = html;
        if (typeof tickerHoverBind === 'function') tickerHoverBind(tbody, '.ticker-badge', null);
        applyDistMA(document.getElementById('scans-table'));
        scanUpdatePriceRows();
        wlRefreshStars();

        allStockRows  = { length: _vsData.length };
        currentStockIndex = -1;
    }

    function _doRenderScans() {
        _renderScansTimer    = null;
        _vsRenderedStart     = -1;
        _vsRenderedEnd       = -1;

        var stocks = getAllStocks();
        var filtered = activeScan
            ? stocks.filter(function(r) { return r[activeScan] === true; })
            : stocks.slice();
        filtered = applyFilters(filtered);

        if (window._scansSearchQuery) {
            var sq = window._scansSearchQuery.toLowerCase();
            filtered = filtered.filter(function(r){ return r.ticker && r.ticker.toLowerCase().includes(sq); });
        }

        _indRankMap = {};
        if (industriesData && industriesData.industries) {
            industriesData.industries.forEach(function(i){ _indRankMap[i.industry] = i.rank || 9999; });
        }

        var _mOpen = wlIsMarketOpen();
        filtered.sort(function(a, b) {
            var key = scansSortState.by;
            var av, bv;
            if (key === 'symbol')   { av = a.ticker;    bv = b.ticker;    return av < bv ? scansSortState.dir : av > bv ? -scansSortState.dir : 0; }
            if (key === 'sector')   { av = a.sector;    bv = b.sector;    return av < bv ? scansSortState.dir : av > bv ? -scansSortState.dir : 0; }
            if (key === 'industry') { av = _indRankMap[a.industry] || 9999; bv = _indRankMap[b.industry] || 9999; return (av - bv) * scansSortState.dir; }
            if (key === 'rs')                   { av = a.Percentile;          bv = b.Percentile; }
            else if (key === 'weighted_rs_pct') { av = a.weighted_rs_pct;     bv = b.weighted_rs_pct; }
            else if (key === 'chg')      { var la=scanLivePrices[a.ticker],lb=scanLivePrices[b.ticker]; av=(_mOpen&&la&&la.price&&la.prevClose)?la.price-la.prevClose:(a.price!=null&&a.daily!=null?(a.price/(1+a.daily/100))*(a.daily/100):null); bv=(_mOpen&&lb&&lb.price&&lb.prevClose)?lb.price-lb.prevClose:(b.price!=null&&b.daily!=null?(b.price/(1+b.daily/100))*(b.daily/100):null); }
            else if (key === 'price')    { var la=scanLivePrices[a.ticker],lb=scanLivePrices[b.ticker]; av=(la&&la.price)?la.price:a.price; bv=(lb&&lb.price)?lb.price:b.price; }
            else if (key === 'daily')    { var la=scanLivePrices[a.ticker],lb=scanLivePrices[b.ticker]; av=(_mOpen&&la&&la.price&&la.prevClose)?((la.price-la.prevClose)/la.prevClose)*100:a.daily; bv=(_mOpen&&lb&&lb.price&&lb.prevClose)?((lb.price-lb.prevClose)/lb.prevClose)*100:b.daily; }
            else if (key === '1w')       { av = a['1w'];     bv = b['1w']; }
            else if (key === '1m')       { av = a['1m'];     bv = b['1m']; }
            else if (key === '3m')       { av = a['3m'];     bv = b['3m']; }
            else if (key === 'ytd')      { av = a['1y'];     bv = b['1y']; }
            else if (key === 'vs_spy')   { av = a.vs_spy;    bv = b.vs_spy; }
            else if (key === 'vs_spy_3m'){ av = a.vs_spy_3m; bv = b.vs_spy_3m; }
            else if (key === 'avg_vol')  { av = a.AvgVol50;  bv = b.AvgVol50; }
            else if (key === 'pct_52wk') { av = a.PctFrom52WkHigh; bv = b.PctFrom52WkHigh; }
            else if (key === 'adr_pct')  { av = a.adr_pct;   bv = b.adr_pct; }
            else if (key === 'cr')       { av = a.cr;        bv = b.cr; }
            else if (key === 'dist_ma')  { av = a.dist_ma ? a.dist_ma[activeMAType+activeMALength] : null; bv = b.dist_ma ? b.dist_ma[activeMAType+activeMALength] : null; }
            else if (key === 'eps_this_y_pct')    { av = a.eps_this_y_pct;    bv = b.eps_this_y_pct; }
            else if (key === 'eps_next_y_pct')    { av = a.eps_next_y_pct;    bv = b.eps_next_y_pct; }
            else if (key === 'eps_next_5y_pct')   { av = a.eps_next_5y_pct;   bv = b.eps_next_5y_pct; }
            else if (key === 'eps_qoq_pct')       { av = a.eps_qoq_pct;       bv = b.eps_qoq_pct; }
            else if (key === 'sales_qoq_pct')     { av = a.sales_qoq_pct;     bv = b.sales_qoq_pct; }
            else if (key === 'profit_margin_pct') { av = a.profit_margin_pct; bv = b.profit_margin_pct; }
            else { av = null; bv = null; }
            if (av == null && bv == null) return 0;
            if (av == null) return 1;
            if (bv == null) return -1;
            return (av - bv) * scansSortState.dir;
        });

        _vsData = filtered;
        scansMcTickers = filtered.map(function(r) { return r.ticker; });
        document.getElementById('scans-result-count').textContent = filtered.length + ' stocks';
        buildScansHeader();

        if (!filtered.length) {
            document.getElementById('scans-tbody').innerHTML =
                '<tr><td colspan="24" style="padding:40px;text-align:center;color:#484f58;font-size:0.935em;">No results</td></tr>';
            allStockRows = [];
            currentStockIndex = -1;
            return;
        }

        var wrap = document.querySelector('#scans-table-view .stocks-table-wrap');
        var _keepScroll = _scanPreserveScroll;
        _scanPreserveScroll = false;
        var _scrollTop = _keepScroll && wrap ? wrap.scrollTop : 0;
        if (!_keepScroll && wrap) wrap.scrollTop = 0;
        _vsRenderWindow(_scrollTop);

        var firstRow = document.querySelector('#scans-tbody .stock-row');
        if (firstRow && firstRow.offsetHeight > 0) _vsRowHeight = firstRow.offsetHeight;

        if (!_vsScrollBound && wrap) {
            _vsScrollBound = true;
            wrap.addEventListener('scroll', function() {
                _vsRenderWindow(wrap.scrollTop);
            }, { passive: true });
        }

        var scansTbody = document.getElementById('scans-tbody');
        scansTbody.onclick = function(e) {
            var row = e.target.closest('.stock-row');
            if (!row || e.target.classList.contains('scan-add-btn')) return;
            scansTbody.querySelectorAll('.stock-row.active').forEach(function(r){ r.classList.remove('active'); });
            row.classList.add('active');
            var topSpacer = document.getElementById('vs-top-spacer');
            var topRows   = topSpacer ? Math.round(topSpacer.offsetHeight / _vsRowHeight) : 0;
            var visRows   = Array.from(scansTbody.querySelectorAll('.stock-row'));
            currentStockIndex = topRows + visRows.indexOf(row);
            allStockRows = { length: _vsData.length };
        };
        scansTbody.oncontextmenu = function(e) {
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

        // On a live-refilter render, skip chart rebuild and price fetch to break
        // the feedback loop: scanFetchPrices → renderScans → scanFetchPrices → …
        // Chart prices are already kept current by _updateMcLiveCandle in scanUpdatePriceRows.
        var _isLiveRefilter = _scanLiveRefilterRender;
        _scanLiveRefilterRender = false;
        if (scansMultichartActive && !_isLiveRefilter) renderScansMc();
        // Refresh live prices for the newly rendered set
        if (currentView === 'scans' && !_isLiveRefilter) scanFetchPrices();
        if (typeof alStampBadges === 'function') alStampBadges();
    }

    // ── Scans live prices ─────────────────────────────────────────────────
    window.sfRefreshScan = function() {
        var btn = document.getElementById('scans-refresh-btn');
        if (btn) btn.classList.add('spinning');
        renderScans();
    };

    // Generation counter: each scanFetchPrices() call bumps this. In-flight
    // requests from a superseded call check their captured generation before
    // applying results, so stale data (e.g. from before a filter change)
    // never overwrites fresher state. _scanFetchActive guards the periodic
    // timer specifically, so it doesn't stack a fresh full wave on top of
    // one still in progress; filter/render-triggered calls always proceed
    // regardless, since user-driven changes should take priority.
    var _scanFetchGen = 0;
    var _scanFetchActive = false;
    var SCAN_FETCH_CONCURRENCY = 2; // max simultaneous 30-ticker batches in flight

    function scanFetchPrices() {
        var tickers = _vsData.map(function(r) { return r.ticker; });
        if (!tickers.length) return;

        var myGen = ++_scanFetchGen;
        _scanFetchActive = true;

        var batches = [];
        // 50 — matches the Worker's cap (raised from 30, the Yahoo-era
        // CPU-budget limit, but deliberately stopped short of the confirmed
        // 20 req/sec ceiling's full headroom since Questrade doesn't document
        // a max ids-per-call limit and this hasn't been verified live). MUST
        // stay in sync with the Worker's cap and state.js/watchlists.js/
        // alerts.js's own constants — a mismatch doesn't error, it just
        // silently returns fewer quotes.
        for (var i = 0; i < tickers.length; i += 50) batches.push(tickers.slice(i, i + 50));

        var idx = 0;

        function runRound() {
            if (myGen !== _scanFetchGen) return; // superseded by a newer call; stop silently

            if (_mcFsIsOpen()) { setTimeout(runRound, 1000); return; } // fullscreen chart open — pause the drain, recheck shortly

            if (idx >= batches.length) {
                if (myGen === _scanFetchGen) _scanFetchActive = false;
                var _refreshBtn = document.getElementById('scans-refresh-btn');
                if (_refreshBtn) _refreshBtn.classList.remove('spinning');
                return;
            }

            var slice = batches.slice(idx, idx + SCAN_FETCH_CONCURRENCY);
            idx += SCAN_FETCH_CONCURRENCY;

            var roundPromises = slice.map(function(batch) {
                var url = WL_PROXY + '?action=quotes_batch&tickers=' + batch.map(encodeURIComponent).join(',');
                return fetch(url).then(function(r) { return r.ok ? r.json() : null; }).then(function(data) {
                    if (myGen !== _scanFetchGen) return; // stale; discard results
                    if (!data || !data.quotes) return;
                    data.quotes.forEach(function(q) {
                        if (q && q.ticker && q.price) {
                            // prevClose now comes from the daily snapshot's preserved
                            // close (tickerMap[ticker]._snapPrice), not the Worker
                            // response — Questrade quotes don't include one.
                            var snapRow   = tickerMap && tickerMap[q.ticker];
                            var prevClose = snapRow ? snapRow._snapPrice : null;
                            scanLivePrices[q.ticker] = { price: q.price, prevClose: prevClose || null, dayHigh: q.dayHigh || null, dayLow: q.dayLow || null };
                            var dataRow = _vsData.find(function(r) { return r.ticker === q.ticker; });
                            if (dataRow) {
                                if (q.dayHigh && q.dayLow && q.dayHigh > q.dayLow)
                                    dataRow.cr = ((q.price - q.dayLow) / (q.dayHigh - q.dayLow)) * 100;
                                // Only overwrite the real daily-change value
                                // (from build_data.py) with a live delta while
                                // the market is actually open. Outside market
                                // hours the "live" price is just the last
                                // traded price — identical to prevClose — which
                                // would silently clobber the correct stored
                                // value with a spurious 0.
                                if (wlIsMarketOpen() && prevClose && prevClose > 0)
                                    dataRow.daily = ((q.price - prevClose) / prevClose) * 100;
                            }
                        }
                    });
                    if (myGen !== _scanFetchGen) return;
                    scanUpdatePriceRows();
                    // Re-apply intraday-sensitive filters now that live data is available.
                    // Stocks that were filtered out using snapshot values get a second pass.
                    var _intradayActive = sfRows.some(function(f) {
                        return (f.type === 'perf' && (f.perfTf || '1d') === '1d') ||
                               (f.type === 'cr'   && (!f.crTf || f.crTf === 'd'));
                    });
                    if (_intradayActive && !_scanLiveRefilterScheduled) {
                        _scanLiveRefilterScheduled = true;
                        setTimeout(function() {
                            _scanLiveRefilterScheduled = false;
                            _scanPreserveScroll = true;
                            _scanLiveRefilterRender = true;
                            renderScans();
                        }, 200);
                    }
                }).catch(function() {});
            });

            Promise.all(roundPromises).then(function() {
                runRound();
            });
        }

        runRound();
    }

    function scanUpdatePriceRows() {
        var marketOpen = wlIsMarketOpen();
        document.querySelectorAll('#scans-tbody .stock-row').forEach(function(tr) {
            var ticker = tr.getAttribute('data-symbol');
            var live   = scanLivePrices[ticker];
            if (!live || !live.price) return;
            var price     = live.price;
            var prevClose = live.prevClose;
            var chgAbs    = (marketOpen && prevClose && prevClose > 0) ? price - prevClose : null;
            var chgPct    = (marketOpen && prevClose && prevClose > 0) ? ((price - prevClose) / prevClose) * 100 : null;
            // Outside market hours (or no live prevClose), fall back to the
            // row's real daily-change value instead of showing blank/zero —
            // this function previously had no fallback at all.
            if (chgPct == null) {
                var dataRow = _vsData.find(function(r) { return r.ticker === ticker; });
                if (dataRow && dataRow.daily != null) {
                    chgPct = dataRow.daily;
                    chgAbs = dataRow.price ? (dataRow.price / (1 + dataRow.daily / 100)) * (dataRow.daily / 100) : null;
                }
            }
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
                var snapDistAll = snapRow ? snapRow.dist_ma : null;
                var dmaKey      = activeMAType + activeMALength;
                var snapDist    = snapDistAll ? snapDistAll[dmaKey] : null;
                if (snapPrice && snapDist != null) {
                    var ma       = snapPrice / (1 + snapDist / 100);
                    var liveDist = (price - ma) / ma * 100;
                    var ldCl     = liveDist > 0 ? 'up' : liveDist < 0 ? 'down' : '';
                    distCell.innerHTML = '<span class="' + ldCl + '">' + fmt(liveDist, 2, '%') + '</span>';
                    tr.setAttribute('data-dist_ma', liveDist);
                }
            }
            // ── Live multichart candle update ─────────────────────────────
            _updateMcLiveCandle(ticker, price, live.dayHigh, live.dayLow, scansMcWidgets);
        });
    }

    function scanStartPricePolling() {
        if (scanPriceTimer) clearInterval(scanPriceTimer);
        if (!wlIsMarketOpen()) return;
        scanPriceTimer = setInterval(function() {
            if (currentView !== 'scans') { scanStopPricePolling(); return; }
            if (!wlIsMarketOpen()) { scanStopPricePolling(); return; }
            if (_scanFetchActive) return; // previous cycle still running; skip this tick rather than stacking another wave
            scanFetchPrices();
        }, 10 * 1000);
    }

    function scanStopPricePolling() {
        if (scanPriceTimer) { clearInterval(scanPriceTimer); scanPriceTimer = null; }
        // runRound() already checks (myGen !== _scanFetchGen) before each
        // round, but nothing was bumping _scanFetchGen when leaving the view
        // — so a drain in progress just kept running to completion regardless.
        // Bumping it here means that guard actually fires on the next round.
        _scanFetchGen++;
        _scanFetchActive = false;
    }

    // ── Scans search filter ───────────────────────────────────────────────
    function filterScansTable(q) {
        // With virtual scroll, filter by re-rendering against the data array
        window._scansSearchQuery = (q || '').trim();
        renderScans();
    }

    // ── Scans Multichart ─────────────────────────────────────────────────
    window.toggleScansMultichart = function() {
        scansMultichartActive = !scansMultichartActive;
        document.getElementById('scans-table-view').style.display      = scansMultichartActive ? 'none' : 'flex';
        document.getElementById('scans-multichart-view').style.display = scansMultichartActive ? 'flex' : 'none';
        var btn = document.getElementById('scans-multichart-toggle-btn');
        btn.style.background  = scansMultichartActive ? '#1f3a5c' : '';
        btn.style.borderColor = scansMultichartActive ? '#388bfd' : '';
        btn.style.color       = scansMultichartActive ? '#58a6ff' : '';
        if (scansMultichartActive) renderScansMc();
    };

    window.setScansMcTf = function(tf) {
        scansMcTimeframe = tf;
        document.querySelectorAll('#scans-multichart-view .mc-tf-btn').forEach(function(b){
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        renderScansMc();
    };

    window.setScansMcCols = function(n) {
        scansMcCols = n;
        document.querySelectorAll('#scans-multichart-view .mc-col-btn').forEach(function(b){
            b.classList.toggle('active', +b.getAttribute('data-cols') === n);
        });
        document.getElementById('scans-multichart-grid').style.gridTemplateColumns = 'repeat(' + n + ', 1fr)';
    };

    function renderScansMc() {
        var grid = document.getElementById('scans-multichart-grid');
        _buildLwMcGrid(grid, scansMcTickers, scansMcTimeframe, scansMcCols, scansMcWidgets, 'scans');
    }

    function wlSearchQuery(q) {
        var scroll = document.getElementById('wl-list-scroll');
        if (!q || !q.trim()) {
            wlRender();
            return;
        }
        q = q.trim().toUpperCase();
        var all   = wlGetAll();
        var order = wlGetOrder();

        // Gather matches across all lists
        var matches = [];
        order.forEach(function(listName) {
            var tickers = all[listName] || [];
            tickers.forEach(function(t) {
                if (t.toUpperCase().indexOf(q) !== -1) {
                    matches.push({ ticker: t, list: listName });
                }
            });
        });

        if (!matches.length) {
            scroll.innerHTML = '<div class="wl-empty-state">No matches for "' + esc(q) + '"</div>';
            return;
        }

        var html = '<div class="wl-col-hdr">';
        html += '<span class="wl-c-sym">Symbol</span>';
        html += '<span style="flex:1;font-size:0.792em;font-weight:600;color:#484f58;text-transform:uppercase;letter-spacing:0.04em;">List</span>';
        html += '</div>';

        var marketOpen = wlIsMarketOpen();
        matches.forEach(function(m) {
            var live = wlLivePrices[m.ticker];
            var sd   = wlLookupStock(m.ticker);
            var price, dayVal;
            if (live) {
                price  = live.price;
                dayVal = (marketOpen && live.prevClose) ? ((live.price - live.prevClose) / live.prevClose) * 100 : (sd ? sd.daily : null);
            } else {
                price  = sd && sd.price != null ? sd.price : null;
                dayVal = sd ? sd.daily : null;
            }
            var priceStr = price != null ? price.toFixed(2) : '—';
            var chgpStr  = dayVal != null ? (dayVal >= 0 ? '+' : '') + dayVal.toFixed(2) + '%' : '';
            var cl       = dayVal == null ? '' : dayVal > 0 ? 'up' : dayVal < 0 ? 'down' : '';

            html += '<div class="wl-ticker-row" data-wl-ticker="' + esc(m.ticker) + '" data-wl-list="' + esc(m.list) + '" style="cursor:pointer;">';
            html += '<span class="wl-c-sym">' + esc(m.ticker) + '</span>';
            html += '<span style="flex:1;font-size:0.858em;color:#6e7681;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(m.list) + '</span>';
            html += '<span class="wl-c-last" style="margin-left:auto;">' + priceStr + '</span>';
            html += '<span class="wl-c-chgp ' + cl + '" style="width:52px;text-align:right;font-size:0.858em;">' + chgpStr + '</span>';
            html += '</div>';
        });

        scroll.innerHTML = html;

        // Wire click — switch to that list and select the ticker
        scroll.querySelectorAll('.wl-ticker-row').forEach(function(row) {
            row.addEventListener('click', function() {
                var ticker   = row.getAttribute('data-wl-ticker');
                var listName = row.getAttribute('data-wl-list');
                document.getElementById('search-input').value = '';
                wlSetLastList(listName);
                wlLivePrices = {};
                wlRender();
                wlStartPricePolling();
                wlSelectTicker(ticker);
            });
        });
    }

