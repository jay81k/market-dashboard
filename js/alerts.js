    // ── PRICE ALERTS ─────────────────────────────────────────────────────────
    var alertsList       = [];   // [{ticker, condition:'above'|'below', price, addedAt}]
    var alSortKey = 'away';  // 'away' | 'added' | 'chgpct' | null
    var alSortDir = 'asc';   // 'asc' | 'desc'
    var alertFiredList   = [];   // [{ticker, condition, alertPrice, hitPrice, firedAt, dismissed}]
    var alertPrices      = {};   // {ticker: latestPrice}
    var alertPrevClose   = {};   // {ticker: prevClosePrice}
    var alertEstimatedMAs = {};  // {"ticker_maKey": estimatedMAValue} derived from snapshot
    var alertPriceTimer  = null;
    var alertOpenTimer   = null;  // setTimeout handle for market-open retry
    var _alertFiredSess  = {};   // prevents re-firing in same session
    var _alActiveTicker  = null; // ticker whose rows are currently selected in the alert list
    var _alReturnState   = null;  // saved state to restore after adding alert from chart modal
    var _scanReturnState = null;  // saved state to restore when returning to scans/stocks view
    var _alEditIdx       = null;  // index of alert being edited, null when adding new

    var LS_AL_KEY       = 'price_alerts_local';
    var LS_AL_FIRED_KEY = 'alerts_fired_local';

    var _alLoaded = false; // guard: once the user has mutated alerts, ignore any late alLoad responses

    function alLoad() {
        // Alert search bar
        var _alSrch = document.getElementById('al-search-input');
        var _alSrchClr = document.getElementById('al-search-clear-btn');
        if (_alSrch) {
            _alSrch.addEventListener('input', function() {
                _alSearchTerm = _alSrch.value;
                if (_alSrchClr) _alSrchClr.classList.toggle('visible', !!_alSearchTerm);
                applyAlFilter();
            });
            if (_alSrchClr) {
                _alSrchClr.addEventListener('click', function() {
                    _alSrch.value = '';
                    _alSearchTerm = '';
                    _alSrchClr.classList.remove('visible');
                    applyAlFilter();
                });
            }
        }
        Promise.all([
            kvGet('price_alerts'),
            kvGet('alerts_fired')
        ]).then(function(r) {
            // If the user already added/deleted an alert before KV responded, don't overwrite their changes
            if (_alLoaded) return;
            _alLoaded = true;
            // KV first, localStorage fallback
            var rawAlerts = r[0] || localStorage.getItem(LS_AL_KEY);
            var rawFired  = r[1] || localStorage.getItem(LS_AL_FIRED_KEY);
            try { alertsList     = rawAlerts ? JSON.parse(rawAlerts) : []; } catch(e) { alertsList = []; }
            try { alertFiredList = rawFired  ? JSON.parse(rawFired)  : []; } catch(e) { alertFiredList = []; }
            // Mirror KV data to localStorage so fallback stays fresh
            if (r[0]) { try { localStorage.setItem(LS_AL_KEY,       r[0]); } catch(e) {} }
            if (r[1]) { try { localStorage.setItem(LS_AL_FIRED_KEY, r[1]); } catch(e) {} }
            alertFiredList.forEach(function(f) {
                _alertFiredSess[f.ticker + '_' + f.alertPrice + '_' + f.condition] = true;
            });
            alUpdateBadge();
            alStartBackgroundPolling();
            renderHistory();
            if (currentView === 'alerts') renderAlerts();
            alStampBadges();
        }).catch(function() {
            if (_alLoaded) return;
            _alLoaded = true;
            // KV totally unavailable — load from localStorage
            try { alertsList     = JSON.parse(localStorage.getItem(LS_AL_KEY)       || '[]'); } catch(e) { alertsList = []; }
            try { alertFiredList = JSON.parse(localStorage.getItem(LS_AL_FIRED_KEY) || '[]'); } catch(e) { alertFiredList = []; }
            alUpdateBadge();
            alStartBackgroundPolling();
            renderHistory();
            if (currentView === 'alerts') renderAlerts();
            alStampBadges();
        });
    }

    function alSave() {
        _alLoaded = true; // mark as user-owned so any late alLoad response won't overwrite
        var str = JSON.stringify(alertsList);
        kvSet('price_alerts', str);
        try { localStorage.setItem(LS_AL_KEY, str); } catch(e) {}
        alStampBadges();
    }

    var _AL_BELL_SVG = '<svg width="8" height="9" viewBox="0 0 8 9" fill="none" style="flex-shrink:0;display:block;"><path d="M4 1a2 2 0 0 1 2 2v1.5l.8 1H1.2L2 4.5V3A2 2 0 0 1 4 1zm-1 5.5h2" stroke="#e3852b" stroke-width="1.1" stroke-linecap="round"/></svg>';
    function _alMakePill(ticker, count) {
        var pill = document.createElement('span');
        pill.className = 'al-ticker-pill';
        pill.title = count > 1 ? count + ' alerts' : 'Alert set';
        pill.innerHTML = _AL_BELL_SVG;
        pill.addEventListener('click', function(e) { e.stopPropagation(); alGoToTicker(ticker); });
        return pill;
    }
    window.alStampBadges = function() {
        // For table rows: insert pill as sibling after the badge element
        var _stampSibling = function(el, ticker) {
            var sib = el.nextSibling;
            while (sib && sib.classList && sib.classList.contains('al-ticker-pill')) {
                var rem = sib; sib = sib.nextSibling; rem.parentNode.removeChild(rem);
            }
            var count = alertsList.filter(function(a) { return a.ticker === ticker; }).length;
            if (count > 0) el.parentNode.insertBefore(_alMakePill(ticker, count), el.nextSibling);
        };
        // For watchlist sym cells: append pill inside the element so it doesn't affect flex layout
        var _stampInside = function(el, ticker) {
            Array.from(el.querySelectorAll('.al-ticker-pill')).forEach(function(p) { p.parentNode.removeChild(p); });
            var count = alertsList.filter(function(a) { return a.ticker === ticker; }).length;
            if (count > 0) el.insertBefore(_alMakePill(ticker, count), el.firstChild);
        };
        // For side panel: append pill inside element to the right of the text
        var _stampInsideRight = function(el, ticker) {
            Array.from(el.querySelectorAll('.al-ticker-pill')).forEach(function(p) { p.parentNode.removeChild(p); });
            var count = alertsList.filter(function(a) { return a.ticker === ticker; }).length;
            if (count > 0) el.appendChild(_alMakePill(ticker, count));
        };
        document.querySelectorAll('.ticker-badge').forEach(function(el) {
            _stampSibling(el, el.textContent.trim());
        });
        document.querySelectorAll('.wl-c-sym').forEach(function(el) {
            var t = el.textContent.trim();
            if (t && t !== 'Symbol') _stampInside(el, t);
        });
        document.querySelectorAll('.snp-ticker').forEach(function(el) {
            _stampInsideRight(el, el.textContent.trim());
        });
    };
    window.alGoToTicker = function(ticker) {
        _alActiveTicker = ticker;
        // Capture state BEFORE closing anything so it can be restored when returning
        var _modalOpen = document.getElementById('mc-fullscreen-overlay').classList.contains('open');
        var _snpOpen   = document.getElementById('scan-nav-panel').classList.contains('snp-open');
        var _openSym   = _modalOpen ? (document.getElementById('mc-fullscreen-sym').textContent.trim() || null) : null;
        var _openSec   = (_openSym && tickerMap && tickerMap[_openSym]) ? (tickerMap[_openSym].sector   || '') : '';
        var _openInd   = (_openSym && tickerMap && tickerMap[_openSym]) ? (tickerMap[_openSym].industry || '') : '';
        if (_modalOpen || _snpOpen) {
            _scanReturnState = null;  // prevent stale _scanReturnState from colliding with _alReturnState on return
            _alReturnState = { view: currentView, ticker: _openSym, sector: _openSec, industry: _openInd, snpOpen: _snpOpen };
        }
        if (_modalOpen) { _alGoToTickerClosing = true; closeChartModal(); _alGoToTickerClosing = false; }
        snpHide();
        showView('alerts');
        setTimeout(function() {
            var listEl = document.getElementById('al-list');
            if (!listEl) return;
            var rows = listEl.querySelectorAll('.al-row');
            // Clear any existing active selection
            rows.forEach(function(r) { r.classList.remove('al-row-active'); });
            var firstMatch = null;
            rows.forEach(function(row) {
                var tEl = row.querySelector('.al-col-ticker');
                if (tEl && tEl.textContent.trim() === ticker) {
                    row.classList.add('al-row-active');
                    if (!firstMatch) firstMatch = row;
                }
            });
            if (firstMatch) firstMatch.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }, 80);
    };

    function alSaveFired() {
        var str = JSON.stringify(alertFiredList.slice(0, 100));
        kvSet('alerts_fired', str);
        try { localStorage.setItem(LS_AL_FIRED_KEY, str); } catch(e) {}
    }

    function alUpdateBadge() {
        var n = alertFiredList.filter(function(f) { return !f.dismissed; }).length;
        var el = document.getElementById('al-nav-badge');
        if (!el) return;
        el.textContent = n || '';
        el.classList.toggle('visible', n > 0);
    }

    function alMsUntilMarketOpen() {
        var now  = new Date();
        var et   = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
        var next = new Date(et);
        next.setHours(9, 30, 0, 0);
        if (et >= next) next.setDate(next.getDate() + 1);
        while (next.getDay() === 0 || next.getDay() === 6) next.setDate(next.getDate() + 1);
        next.setHours(9, 30, 0, 0);
        var etOffset = et.getTime() - now.getTime();
        return Math.max(next.getTime() - etOffset - now.getTime(), 0);
    }

    function alStartBackgroundPolling() {
        if (alertPriceTimer) { clearInterval(alertPriceTimer); alertPriceTimer = null; }
        if (alertOpenTimer)  { clearTimeout(alertOpenTimer);   alertOpenTimer  = null; }
        if (!alertsList.length && !alertFiredList.length) return;
        alFetchPrices();
        if (!wlIsMarketOpen()) {
            alertOpenTimer = setTimeout(alStartBackgroundPolling, alMsUntilMarketOpen());
            return;
        }
        alertPriceTimer = setInterval(function() {
            if (!wlIsMarketOpen()) {
                clearInterval(alertPriceTimer); alertPriceTimer = null;
                alertOpenTimer = setTimeout(alStartBackgroundPolling, alMsUntilMarketOpen());
                return;
            }
            alFetchPrices();
        }, 60 * 1000);
    }

    function alFetchPrices() {
        var activeTickers  = alertsList.map(function(a) { return a.ticker; });
        var historyTickers = alertFiredList.map(function(f) { return f.ticker; });
        var tickers = activeTickers.concat(historyTickers)
            .filter(function(v, i, arr) { return arr.indexOf(v) === i; });
        if (!tickers.length) return Promise.resolve();
        // Batch into chunks of 50 to avoid URL length limits
        var batches = [];
        for (var i = 0; i < tickers.length; i += 50) batches.push(tickers.slice(i, i + 50));
        return Promise.all(batches.map(function(batch) {
            var url = WL_PROXY + '?action=quotes_batch&tickers=' + batch.map(encodeURIComponent).join(',');
            return fetch(url).then(function(r) { return r.ok ? r.json() : null; }).catch(function() { return null; });
        })).then(function(results) {
            results.forEach(function(data) {
                if (!data || !data.quotes) return;
                data.quotes.forEach(function(q) {
                    if (q && q.ticker && q.price) {
                        alertPrices[q.ticker]    = q.price;
                        alertPrevClose[q.ticker] = q.prevClose || null;
                    }
                });
            });
            alUpdateEstimatedMAs();
            alCheckTriggers();
            if (currentView === 'alerts') renderAlerts();
        }).catch(function() {});
    }

    function alPlayAlert() {
        try {
            var ctx  = new (window.AudioContext || window.webkitAudioContext)();
            // Three sharp descending beeps
            [0, 0.18, 0.36].forEach(function(startTime) {
                var osc  = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type      = 'square';
                osc.frequency.setValueAtTime(880, ctx.currentTime + startTime);
                gain.gain.setValueAtTime(0.35, ctx.currentTime + startTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + startTime + 0.14);
                osc.start(ctx.currentTime + startTime);
                osc.stop(ctx.currentTime + startTime + 0.14);
            });
            setTimeout(function() { ctx.close(); }, 800);
        } catch(e) {}
    }

    function alUpdateEstimatedMAs() {
        if (!snapshot || !snapshot.by_industry) return;
        var maAlerts = alertsList.filter(function(a) { return a.alertType === 'ma'; });
        if (!maAlerts.length) return;
        maAlerts.forEach(function(a) {
            var cacheKey = a.ticker + '_' + a.maKey;
            outer: for (var ind in snapshot.by_industry) {
                var stocks = snapshot.by_industry[ind];
                for (var si = 0; si < stocks.length; si++) {
                    var s = stocks[si];
                    if (s.ticker === a.ticker && s.price != null && s.dist_ma && s.dist_ma[a.maKey] != null) {
                        alertEstimatedMAs[cacheKey] = s.price / (1 + s.dist_ma[a.maKey] / 100);
                        break outer;
                    }
                }
            }
        });
    }

    function alCheckTriggers() {
        var anyFired = false;
        var toRemove = [];
        alertsList.forEach(function(a) {
            var key;
            if (a.alertType === 'macross')
                key = a.ticker + '_macross_' + a.ma1Key + '_' + a.ma2Key + '_' + a.condition;
            else if (a.alertType === 'ma')
                key = a.ticker + '_ma_' + a.maKey + '_' + a.condition;
            else if (a.alertType === 'pattern')
                key = window.alPatternAlertKey(a);
            else if (a.alertType === 'trendline')
                key = a.ticker + '_trendline_' + (a.p1 ? a.p1.unix : '') + '_' + (a.p2 ? a.p2.unix : '') + '_' + a.condition;
            else
                key = a.ticker + '_' + a.price + '_' + a.condition;
            if (_alertFiredSess[key]) return;
            var hit = false;
            var hitVal = null;

            // Find this ticker in snapshot (shared by all MA checks)
            var stockData = null;
            if (snapshot && snapshot.by_industry) {
                outer: for (var ind in snapshot.by_industry) {
                    var stocks = snapshot.by_industry[ind];
                    for (var si = 0; si < stocks.length; si++) {
                        if (stocks[si].ticker === a.ticker) { stockData = stocks[si]; break outer; }
                    }
                }
            }

            if (a.alertType === 'macross') {
                if (!stockData) return;
                var v1 = stockData.ma_val ? stockData.ma_val[a.ma1Key] : null;
                var v2 = stockData.ma_val ? stockData.ma_val[a.ma2Key] : null;
                // Event-based: crossover happened in latest candle
                var xKey = a.ma1Key + '|' + a.ma2Key + '|' + a.condition;
                var eventHit = (stockData.ma_crossovers || []).indexOf(xKey) !== -1;
                // State-based: MA 1 is currently above/below MA 2
                var stateHit = (v1 != null && v2 != null) &&
                    ((a.condition === 'above' && v1 > v2) || (a.condition === 'below' && v1 < v2));
                hit = eventHit || stateHit;
                hitVal = (v1 != null && v2 != null) ? ((v1 - v2) / v2 * 100) : 0;
            } else if (a.alertType === 'ma') {
                var livePrice = alertPrices[a.ticker];
                var estMA     = alertEstimatedMAs[a.ticker + '_' + a.maKey];
                // Event-based: price crossed the MA in the latest candle
                var pxKey    = a.maKey + '|' + a.condition;
                var eventHit = stockData && (stockData.price_ma_crossovers || []).indexOf(pxKey) !== -1;
                if (eventHit) {
                    hit = true;
                    hitVal = livePrice != null ? livePrice : (stockData ? (stockData.price || 0) : 0);
                } else if (livePrice != null && estMA != null) {
                    hitVal = livePrice;
                    hit = (a.condition === 'above' && livePrice >= estMA) ||
                          (a.condition === 'below' && livePrice <= estMA);
                } else {
                    var snapDist = stockData ? (stockData.dist_ma ? stockData.dist_ma[a.maKey] : null) : null;
                    if (snapDist == null) return;
                    hitVal = snapDist;
                    hit = (a.condition === 'above' && snapDist >= 0) ||
                          (a.condition === 'below' && snapDist <= 0);
                }
            } else if (a.alertType === 'rsi14') {
                var snapRsi = stockData ? stockData.rsi14 : null;
                if (snapRsi == null) return;
                hitVal = snapRsi;
                hit = (a.condition === 'above' && snapRsi >= a.price) ||
                      (a.condition === 'below' && snapRsi <= a.price);
            } else if (a.alertType === 'pattern') {
                if (!stockData) return;
                var ptfSuffix = (a.patternTf === 'w') ? '_w' : (a.patternTf === 'm') ? '_m' : '';
                var pKeys = window.alGetPatternKeys(a);
                var triggeredPats = pKeys.filter(function(pk) { return !!stockData[pk + ptfSuffix]; });
                hit = triggeredPats.length > 0;
                hitVal = stockData.price || 0;
            } else if (a.alertType === 'trendline') {
                var tlLivePrice = alertPrices[a.ticker];
                if (tlLivePrice == null || !a.p1 || !a.p2) return;
                var tlNowUnix   = _alTlEvalUnix(a.p1.unix, a.p2.unix);
                var tlLinePrice = _alTrendlinePriceAt(a.p1.unix, a.p1.price, a.p2.unix, a.p2.price, tlNowUnix);
                hitVal = tlLivePrice;
                hit = (a.condition === 'above' && tlLivePrice >= tlLinePrice) ||
                      (a.condition === 'below' && tlLivePrice <= tlLinePrice);
            } else {
                var price = alertPrices[a.ticker];
                if (price == null) return;
                hitVal = price;
                hit = (a.condition === 'above' && price >= a.price) ||
                      (a.condition === 'below' && price <= a.price);
            }
            if (!hit) return;
            _alertFiredSess[key] = true;
            alertFiredList.unshift({
                ticker: a.ticker, condition: a.condition,
                alertPrice: a.alertType === 'trendline' ? null : a.price, hitPrice: hitVal,
                alertType: a.alertType || 'price',
                maKey: a.maKey || null,
                ma1Key: a.ma1Key || null, ma2Key: a.ma2Key || null,
                patternKeys: window.alGetPatternKeys(a), patternTf: a.patternTf || null,
                triggeredPatternKeys: (a.alertType === 'pattern' ? triggeredPats : null),
                p1: a.p1 || null, p2: a.p2 || null,
                name: a.name || '',
                firedAt: new Date().toISOString(), dismissed: false
            });
            toRemove.push(key);
            anyFired = true;
            if (window.Notification && Notification.permission === 'granted') {
                var body;
                if (a.alertType === 'macross') {
                    var dir = a.condition === 'above' ? '▲' : '▼';
                    body = dir + ' ' + a.ma1Key.replace(/([A-Z]+)(\d+)/,'$1 $2') + ' ' + a.condition + ' ' + a.ma2Key.replace(/([A-Z]+)(\d+)/,'$1 $2') + ' · spread ' + (hitVal >= 0 ? '+' : '') + hitVal.toFixed(2) + '%';
                } else if (a.alertType === 'ma') {
                    body = (a.condition === 'above' ? '▲ above ' : '▼ below ') + a.maKey.replace(/([A-Z]+)(\d+)/,'$1 $2') + ' · dist ' + (typeof hitVal === 'number' ? hitVal.toFixed(2) : '—') + '%';
                } else if (a.alertType === 'rsi14') {
                    body = 'RSI ' + (a.condition === 'above' ? '▲' : '▼') + ' ' + a.price + ' · now ' + hitVal.toFixed(1);
                } else if (a.alertType === 'pattern') {
                    var pLabels = triggeredPats.map(function(k){ return (AL_PATTERN_LABELS[k] || k).replace(/_/g,' '); }).join(' + ');
                    body = 'Pattern detected: ' + pLabels + ' (' + (a.patternTf || 'd').toUpperCase() + ')';
                } else if (a.alertType === 'trendline') {
                    body = (a.condition === 'above' ? '▲ above' : '▼ below') + ' trendline · now $' + hitVal.toFixed(2);
                } else {
                }
                new Notification(a.ticker + ' alert triggered', { body: body });
            }
        });
        if (toRemove.length) {
            var removeSet = {};
            toRemove.forEach(function(k) { removeSet[k] = true; });
            alertsList = alertsList.filter(function(a) {
                var k;
                if (a.alertType === 'macross') k = a.ticker + '_macross_' + a.ma1Key + '_' + a.ma2Key + '_' + a.condition;
                else if (a.alertType === 'ma') k = a.ticker + '_ma_' + a.maKey + '_' + a.condition;
                else if (a.alertType === 'pattern') k = window.alPatternAlertKey(a);
                else if (a.alertType === 'trendline') k = a.ticker + '_trendline_' + (a.p1 ? a.p1.unix : '') + '_' + (a.p2 ? a.p2.unix : '') + '_' + a.condition;
                else k = a.ticker + '_' + a.price + '_' + a.condition;
                return !removeSet[k];
            });
            alSave();
        }
        if (anyFired) { alPlayAlert(); alSaveFired(); alUpdateBadge(); renderHistory(); }
    }

    window.alToggleSort = function(key) {
        if (alSortKey === key) {
            alSortDir = alSortDir === 'asc' ? 'desc' : 'asc';
        } else {
            alSortKey = key;
            alSortDir = 'asc';
        }
        // Update header indicators
        ['away','added','chgpct'].forEach(function(k) {
            var el = document.getElementById('al-hdr-' + k);
            if (!el) return;
            el.classList.toggle('sorted', k === alSortKey);
            // Strip old arrow
            el.textContent = k === 'away' ? 'Away' : k === 'added' ? 'Added' : 'Chg%';
            if (k === alSortKey) {
                el.textContent += alSortDir === 'asc' ? ' ↑' : ' ↓';
            }
        });
        renderAlerts();
    };

    function renderAlerts() {
        var listEl      = document.getElementById('al-list');
        var hdrRow      = document.getElementById('al-hdr-row');
        var firedRowsEl = document.getElementById('al-fired-rows');
        var banner      = document.getElementById('al-missed-banner');
        if (!listEl) return;

        // Preserve scroll position — every innerHTML replacement resets scrollTop to 0.
        // This snaps the user to the top on every 60-second price poll AND on the async
        // name-resolution render that fires a few seconds after adding an alert.
        var savedScrollTop = listEl.scrollTop;

        if (banner) banner.style.display = 'none';
        if (firedRowsEl) firedRowsEl.innerHTML = '';

        var countEl = document.getElementById('al-count-label');
        if (countEl) countEl.textContent = alertsList.length + ' active';
        var listBadge = document.getElementById('al-list-badge');
        if (listBadge) listBadge.textContent = alertsList.length;
        var listTabBadge = document.getElementById('al-list-tab-badge');
        if (listTabBadge) listTabBadge.textContent = alertsList.length;

        renderHistory();

        if (!alertsList.length) {
            hdrRow.style.display = 'none';
            listEl.innerHTML = '<div class="al-empty">No alerts set. Click "+ Add alert" to get started.</div>';
            return;
        }

        hdrRow.style.display = 'flex';

        // Sync sort header indicators
        ['away','added','chgpct'].forEach(function(k) {
            var el = document.getElementById('al-hdr-' + k);
            if (!el) return;
            el.classList.toggle('sorted', k === alSortKey);
            el.textContent = k === 'away' ? 'Away' : k === 'added' ? 'Added' : 'Chg%';
            if (k === alSortKey) el.textContent += alSortDir === 'asc' ? ' ↑' : ' ↓';
        });

        // Build indexed copy for display, then sort if needed
        var displayList = alertsList.map(function(a, idx) { return { a: a, idx: idx }; });
        if (alSortKey === 'added') {
            displayList.sort(function(x, y) {
                var ta = x.a.addedAt ? new Date(x.a.addedAt).getTime() : Infinity;
                var tb = y.a.addedAt ? new Date(y.a.addedAt).getTime() : Infinity;
                if (ta === Infinity && tb === Infinity) return 0;
                if (ta === Infinity) return 1;
                if (tb === Infinity) return -1;
                return alSortDir === 'asc' ? ta - tb : tb - ta;
            });
        } else if (alSortKey === 'away') {
            displayList.sort(function(x, y) {
                function awayVal(a) {
                    if (a.alertType === 'rsi14' || a.alertType === 'pattern') return Infinity;
                    if (a.alertType === 'trendline') {
                        var tlCurr = alertPrices[a.ticker];
                        if (tlCurr == null || !a.p1 || !a.p2) return Infinity;
                        var tlSortPrice = _alTrendlinePriceAt(a.p1.unix, a.p1.price, a.p2.unix, a.p2.price, _alTlEvalUnix(a.p1.unix, a.p2.unix));
                        return tlSortPrice > 0 ? Math.abs((tlCurr - tlSortPrice) / tlSortPrice * 100) : Infinity;
                    }
                    if (a.alertType === 'macross') {
                        var sd = null;
                        if (snapshot && snapshot.by_industry) {
                            outerS: for (var ind in snapshot.by_industry) {
                                var st = snapshot.by_industry[ind];
                                for (var si = 0; si < st.length; si++) {
                                    if (st[si].ticker === a.ticker) { sd = st[si]; break outerS; }
                                }
                            }
                        }
                        if (!sd || !sd.ma_val) return Infinity;
                        var v1 = sd.ma_val[a.ma1Key], v2 = sd.ma_val[a.ma2Key];
                        return (v1 != null && v2 != null) ? Math.abs((v1 - v2) / v2 * 100) : Infinity;
                    }
                    if (a.alertType === 'ma') {
                        var dist = null;
                        if (snapshot && snapshot.by_industry) {
                            outerM: for (var indM in snapshot.by_industry) {
                                var stM = snapshot.by_industry[indM];
                                for (var siM = 0; siM < stM.length; siM++) {
                                    if (stM[siM].ticker === a.ticker) { dist = stM[siM].dist_ma ? stM[siM].dist_ma[a.maKey] : null; break outerM; }
                                }
                            }
                        }
                        return dist != null ? Math.abs(dist) : Infinity;
                    }
                    var curr = alertPrices[a.ticker];
                    return (curr != null && a.price > 0) ? Math.abs((curr - a.price) / a.price * 100) : Infinity;
                }
                var pa = awayVal(x.a), pb = awayVal(y.a);
                if (pa === Infinity && pb === Infinity) return 0;
                if (pa === Infinity) return 1;
                if (pb === Infinity) return -1;
                return alSortDir === 'asc' ? pa - pb : pb - pa;
            });
        } else if (alSortKey === 'chgpct') {
            displayList.sort(function(x, y) {
                var pa = alertPrices[x.a.ticker], pc_a = alertPrevClose[x.a.ticker];
                var pb = alertPrices[y.a.ticker], pc_b = alertPrevClose[y.a.ticker];
                var va = (pa != null && pc_a != null && pc_a > 0) ? (pa - pc_a) / pc_a * 100 : null;
                var vb = (pb != null && pc_b != null && pc_b > 0) ? (pb - pc_b) / pc_b * 100 : null;
                if (va === null && vb === null) return 0;
                if (va === null) return 1;
                if (vb === null) return -1;
                return alSortDir === 'asc' ? va - vb : vb - va;
            });
        }

        var AL_PATTERN_LABELS = { inside_day: 'Inside Day', double_inside_day: 'Double Inside Day', bullish_outside: 'Bullish Outside', bearish_outside: 'Bearish Outside', hammer: 'Hammer', bullish_reversal_bar: 'Bullish Reversal Bar', upside_reversal: 'Upside Reversal', oops_reversal: 'Oops Reversal', pocket_pivot: 'Pocket Pivot' };

        listEl.innerHTML = displayList.map(function(item) {
            var a = item.a, idx = item.idx;
            var key;
            if (a.alertType === 'macross') key = a.ticker + '_macross_' + a.ma1Key + '_' + a.ma2Key + '_' + a.condition;
            else if (a.alertType === 'ma') key = a.ticker + '_ma_' + a.maKey + '_' + a.condition;
            else if (a.alertType === 'pattern') key = window.alPatternAlertKey(a);
            else if (a.alertType === 'trendline') key = a.ticker + '_trendline_' + (a.p1 ? a.p1.unix : '') + '_' + (a.p2 ? a.p2.unix : '') + '_' + a.condition;
            else key = a.ticker + '_' + a.price + '_' + a.condition;
            var fired = !!_alertFiredSess[key];
            var curr  = alertPrices[a.ticker] != null ? alertPrices[a.ticker] : null;
            var name  = (tickerMap && tickerMap[a.ticker] && tickerMap[a.ticker].name) ? tickerMap[a.ticker].name : (a.name || '');
            var condHtml = a.alertType === 'macross'
                ? (a.condition === 'above'
                    ? '<span style="color:#3fb950;">▲ MA above</span>'
                    : '<span style="color:#f85149;">▼ MA below</span>')
                : a.alertType === 'pattern'
                    ? '<span style="color:#8b949e;">⬡ Pattern</span>'
                    : a.alertType === 'trendline'
                        ? (a.condition === 'above'
                            ? '<span style="color:#3fb950;">▲ trendline</span>'
                            : '<span style="color:#f85149;">▼ trendline</span>')
                        : a.isPrevDay === 'high'
                        ? '<span style="color:#3fb950;">▲ PDH</span>'
                        : a.isPrevDay === 'low'
                            ? '<span style="color:#f85149;">▼ PDL</span>'
                            : a.isPrevDay === '52wk-high'
                                ? '<span style="color:#3fb950;">▲ 52WH</span>'
                                : a.isPrevDay === '52wk-low'
                                    ? '<span style="color:#f85149;">▼ 52WL</span>'
                                    : a.condition === 'above'
                                ? '<span style="color:#3fb950;">▲ above</span>'
                                : '<span style="color:#f85149;">▼ below</span>';
            var awayHtml;
            if (a.alertType === 'macross') {
                var mcStockData = null;
                if (snapshot && snapshot.by_industry) {
                    outerMC: for (var indMC in snapshot.by_industry) {
                        var stMC = snapshot.by_industry[indMC];
                        for (var siMC = 0; siMC < stMC.length; siMC++) {
                            if (stMC[siMC].ticker === a.ticker) { mcStockData = stMC[siMC]; break outerMC; }
                        }
                    }
                }
                if (fired || !mcStockData || !mcStockData.ma_val) {
                    awayHtml = '<div class="al-col-away">—</div>';
                } else {
                    var mcV1 = mcStockData.ma_val[a.ma1Key];
                    var mcV2 = mcStockData.ma_val[a.ma2Key];
                    if (mcV1 == null || mcV2 == null) {
                        awayHtml = '<div class="al-col-away">—</div>';
                    } else {
                        var spread = ((mcV1 - mcV2) / mcV2 * 100);
                        var spreadAbs = Math.abs(spread);
                        var awayCls = spreadAbs < 1 ? ' imminent' : spreadAbs < 5 ? ' close' : '';
                        var spreadStr = (spread >= 0 ? '+' : '') + spread.toFixed(1) + '%';
                        awayHtml = '<div class="al-col-away' + awayCls + '">' + spreadStr + '</div>';
                    }
                }
            } else if (a.alertType === 'ma') {
                var snapDistMA = null;
                if (snapshot && snapshot.by_industry) {
                    outerMA: for (var indMA in snapshot.by_industry) {
                        var stMA = snapshot.by_industry[indMA];
                        for (var siMA = 0; siMA < stMA.length; siMA++) {
                            if (stMA[siMA].ticker === a.ticker) { snapDistMA = stMA[siMA].dist_ma ? stMA[siMA].dist_ma[a.maKey] : null; break outerMA; }
                        }
                    }
                }
                if (fired || snapDistMA == null) {
                    awayHtml = '<div class="al-col-away">—</div>';
                } else {
                    var maDist = Math.abs(snapDistMA);
                    var awayCls = maDist < 1 ? ' imminent' : maDist < 5 ? ' close' : '';
                    awayHtml = '<div class="al-col-away' + awayCls + '">' + maDist.toFixed(1) + '%</div>';
                }
            } else if (a.alertType === 'rsi14') {
                var snapRsi2 = null;
                if (snapshot && snapshot.by_industry) {
                    outer4: for (var ind4 in snapshot.by_industry) {
                        var st4 = snapshot.by_industry[ind4];
                        for (var si4 = 0; si4 < st4.length; si4++) {
                            if (st4[si4].ticker === a.ticker) { snapRsi2 = st4[si4].rsi14; break outer4; }
                        }
                    }
                }
                if (fired || snapRsi2 == null) {
                    awayHtml = '<div class="al-col-away">—</div>';
                } else {
                    var rsiDist = Math.abs(snapRsi2 - a.price);
                    var awayCls = rsiDist < 2 ? ' imminent' : rsiDist < 5 ? ' close' : '';
                    awayHtml = '<div class="al-col-away' + awayCls + '">' + rsiDist.toFixed(1) + '</div>';
                }
            } else if (a.alertType === 'pattern') {
                awayHtml = '<div class="al-col-away">—</div>';
            } else if (a.alertType === 'trendline') {
                if (fired || curr == null || !a.p1 || !a.p2) {
                    awayHtml = '<div class="al-col-away">—</div>';
                } else {
                    var tlNow   = _alTlEvalUnix(a.p1.unix, a.p2.unix);
                    var tlPriceNow = _alTrendlinePriceAt(a.p1.unix, a.p1.price, a.p2.unix, a.p2.price, tlNow);
                    var tlPct   = tlPriceNow > 0 ? Math.abs((curr - tlPriceNow) / tlPriceNow * 100) : 0;
                    var tlCls   = tlPct < 1 ? ' imminent' : tlPct < 5 ? ' close' : '';
                    awayHtml    = '<div class="al-col-away' + tlCls + '">' + tlPct.toFixed(1) + '%</div>';
                }
            } else if (fired || curr == null) {
                awayHtml = '<div class="al-col-away">—</div>';
            } else {
                var pct = Math.abs((curr - a.price) / a.price * 100);
                var awayCls = pct < 1 ? ' imminent' : pct < 5 ? ' close' : '';
                awayHtml = '<div class="al-col-away' + awayCls + '">' + pct.toFixed(1) + '%</div>';
            }
            var prevClose = alertPrevClose[a.ticker] || null;
            var chgAbs    = (curr != null && prevClose && prevClose > 0) ? curr - prevClose : null;
            var chgPct    = (chgAbs != null) ? (chgAbs / prevClose * 100) : null;
            var chgCls    = chgAbs == null ? 'flat' : chgAbs > 0 ? 'up' : chgAbs < 0 ? 'dn' : 'flat';
            var chgHtml    = '<div class="al-col-chg '    + chgCls + '">' + (chgAbs  != null ? (chgAbs  >= 0 ? '+' : '') + chgAbs.toFixed(2)  : '—') + '</div>';
            var chgPctHtml = '<div class="al-col-chgpct ' + chgCls + '">' + (chgPct  != null ? (chgPct  >= 0 ? '+' : '') + chgPct.toFixed(2) + '%' : '—') + '</div>';
            var addedHtml;
            if (a.addedAt) {
                var d = new Date(a.addedAt);
                var isToday = d.toDateString() === new Date().toDateString();
                addedHtml = isToday
                    ? d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
                    : d.toLocaleDateString([], {month:'short', day:'numeric'});
            } else { addedHtml = '—'; }
            return '<div class="al-row' + (fired ? ' fired' : '') + '">' +
                '<div class="al-col-ticker al-col-ticker-link" onclick="alTickerClick(\'' + a.ticker + '\')">' + esc(a.ticker) + '</div>' +
                '<div class="al-col-name">' + esc(name) + '</div>' +
                '<div class="al-col-cond">' + condHtml + '</div>' +
                '<div class="al-col-target ' + a.condition + ((a.alertType === 'macross' || a.alertType === 'ma') ? ' ma-label' : '') + '"' + (a.alertType === 'pattern' ? ' data-al-tip="' + window.alGetPatternKeys(a).map(function(k){return AL_PATTERN_LABELS[k]||k;}).join('\n') + '"' : '') + '>' + (a.alertType === 'rsi14' ? 'RSI ' + a.price : a.alertType === 'macross' ? a.ma1Key.replace(/([A-Z]+)(\d+)/,'$1 $2') + ' × ' + a.ma2Key.replace(/([A-Z]+)(\d+)/,'$1 $2') : a.alertType === 'ma' ? a.maKey.replace(/([A-Z]+)(\d+)/,'$1 $2') : a.alertType === 'pattern' ? (function(){ var pk = window.alGetPatternKeys(a); var tf = (a.patternTf||'d').toUpperCase(); if (pk.length === 1) { return '<span class="al-pat-single"><span>' + (AL_PATTERN_LABELS[pk[0]]||pk[0]) + '</span><span class="al-pat-single-tf">' + tf + '</span></span>'; } return '<span class="al-pat-multi"><svg width="16" height="12" viewBox="0 0 16 12" fill="none" style="flex-shrink:0"><polyline points="0,9 3,9 5,3 7,10 9,6 11,7 13,4 16,4" stroke="#a78bfa" stroke-width="1.5" fill="none" stroke-linejoin="round" stroke-linecap="round"/></svg><span class="al-pat-multi-count">' + pk.length + '</span><span class="al-pat-multi-tf">' + tf + '</span></span>'; })() : a.alertType === 'trendline' ? '<span style="display:inline-flex;align-items:center;"><svg width="16" height="12" viewBox="0 0 16 12" fill="none" style="flex-shrink:0"><line x1="1" y1="11" x2="15" y2="1" stroke="#8b949e" stroke-width="1.5" stroke-linecap="round"/><circle cx="2" cy="10.5" r="1.8" fill="#8b949e"/><circle cx="14" cy="1.5" r="1.8" fill="#8b949e"/></svg></span>' : '$' + a.price.toFixed(2)) + '</div>' +
                '<div class="al-col-curr">' + (a.alertType === 'rsi14' ? (function() { var r = null; if (snapshot && snapshot.by_industry) { outer3: for (var i3 in snapshot.by_industry) { var s3 = snapshot.by_industry[i3]; for (var j3=0;j3<s3.length;j3++) { if(s3[j3].ticker===a.ticker){r=s3[j3].rsi14;break outer3;} } } } return r != null ? r.toFixed(1) : '—'; })() : (curr != null ? '$' + curr.toFixed(2) : '—')) + '</div>' +
                chgHtml +
                chgPctHtml +
                awayHtml +
                '<div class="al-col-status">' + (fired ? '<span class="al-pill al-pill-fired">Fired</span>' : '<span class="al-pill al-pill-active">Active</span>') + '</div>' +
                '<div class="al-col-added">' + addedHtml + '</div>' +
                (a.alertType === 'trendline'
                    ? '<div class="al-col-edit" style="visibility:hidden;pointer-events:none;">✎</div>'
                    : '<div class="al-col-edit" onclick="alEditOpen(' + idx + ')" title="Edit">✎</div>') +
                '<div class="al-col-del" onclick="alDeleteConfirm(' + idx + ',\'' + esc(a.ticker) + '\')" title="Remove">×</div>' +
            '</div>';
        }).join('');
        // Restore scroll so price polls and async name updates don't snap to top.
        listEl.scrollTop = savedScrollTop;
        if (typeof tickerHoverBind === 'function') tickerHoverBind(listEl, '.al-col-ticker', null);
        alStampBadges();

        // Re-apply active ticker selection that was set via bell-click (survives re-renders)
        if (_alActiveTicker) {
            listEl.querySelectorAll('.al-row').forEach(function(r) {
                var tEl = r.querySelector('.al-col-ticker');
                if (tEl && tEl.textContent.trim() === _alActiveTicker) r.classList.add('al-row-active');
            });
        }

        // Click a row to select it (clears previous selection)
        listEl.onclick = function(e) {
            var row = e.target.closest('.al-row');
            if (!row) return;
            // Don't steal clicks from edit/delete/ticker-link buttons
            if (e.target.closest('.al-col-edit') || e.target.closest('.al-col-del') || e.target.closest('.al-col-ticker-link')) return;
            var tEl = row.querySelector('.al-col-ticker');
            var ticker = tEl ? tEl.textContent.trim() : null;
            if (!ticker) return;
            _alActiveTicker = ticker;
            listEl.querySelectorAll('.al-row').forEach(function(r) { r.classList.remove('al-row-active'); });
            listEl.querySelectorAll('.al-row').forEach(function(r) {
                var t = r.querySelector('.al-col-ticker');
                if (t && t.textContent.trim() === ticker) r.classList.add('al-row-active');
            });
        };

        // Right-click on any alert row → watchlist / add-alert picker
        listEl.oncontextmenu = function(e) {
            var row = e.target.closest('.al-row');
            if (!row) return;
            var tickerEl = row.querySelector('.al-col-ticker');
            if (!tickerEl) return;
            var ticker = tickerEl.textContent.trim();
            if (!ticker) return;
            e.preventDefault();
            // Always dismiss any open picker first — fakeBtn is a new object each
            // time so the normal btn===btn toggle inside wlOpenPicker never fires.
            wlClosePicker();
            var fakeBtn = {
                getAttribute: function(attr) { return attr === 'data-ticker' ? ticker : null; },
                getBoundingClientRect: function() { return { bottom: e.clientY, top: e.clientY, left: e.clientX }; },
                _wlNoSwitch: true
            };
            wlOpenPicker(fakeBtn, e, false);
        };

        applyAlFilter();
    }

    var _alSearchTerm = '';
    function applyAlFilter() {
        var term = _alSearchTerm.trim().toLowerCase();
        var rows = document.querySelectorAll('#al-list .al-row');
        rows.forEach(function(row) {
            if (!term) { row.style.display = ''; return; }
            var ticker = (row.querySelector('.al-col-ticker') || {}).textContent || '';
            var name   = (row.querySelector('.al-col-name')   || {}).textContent || '';
            row.style.display = (ticker.toLowerCase().indexOf(term) > -1 || name.toLowerCase().indexOf(term) > -1) ? '' : 'none';
        });
    }

    function renderHistory() {
        var listEl = document.getElementById('al-hist-list');
        var badge  = document.getElementById('al-hist-badge');
        var tabBadge = document.getElementById('al-hist-tab-badge');
        if (!listEl) return;
        var count = alertFiredList.length;
        if (badge)    badge.textContent    = count;
        if (tabBadge) tabBadge.textContent = count;
        if (!count) {
            listEl.innerHTML = '<div class="al-hist-empty">No fired alerts yet.</div>';
            return;
        }
        listEl.innerHTML = alertFiredList.map(function(f, idx) {
            var t = new Date(f.firedAt);
            var isToday = t.toDateString() === new Date().toDateString();
            var ts = isToday
                ? t.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
                : t.toLocaleDateString([], {month:'short', day:'numeric'}) + ' ' + t.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
            var AL_HIST_PAT_LABELS = { inside_day: 'Inside Day', double_inside_day: 'Double Inside Day', bullish_outside: 'Bullish Outside', bearish_outside: 'Bearish Outside', hammer: 'Hammer', bullish_reversal_bar: 'Bullish Reversal Bar', upside_reversal: 'Upside Reversal', oops_reversal: 'Oops Reversal', pocket_pivot: 'Pocket Pivot' };
            var condHtml = f.alertType === 'macross'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? '#3fb950' : '#f85149') + ';">' +
                  (f.condition === 'above' ? '▲' : '▼') + ' ' +
                  (f.ma1Key || '').replace(/([A-Z]+)(\d+)/,'$1 $2') + ' × ' +
                  (f.ma2Key || '').replace(/([A-Z]+)(\d+)/,'$1 $2') + '</span>'
                : f.alertType === 'ma'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? '#3fb950' : '#f85149') + ';">' + (f.condition === 'above' ? '▲' : '▼') + ' ' + (f.maKey || '').replace(/([A-Z]+)(\d+)/,'$1 $2') + '</span>'
                : f.alertType === 'trendline'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? '#3fb950' : '#f85149') + ';">' + (f.condition === 'above' ? '▲' : '▼') + ' trendline</span>'
                : f.alertType === 'pattern'
                ? (function() {
                    var pKeys = (f.triggeredPatternKeys && f.triggeredPatternKeys.length)
                        ? f.triggeredPatternKeys
                        : (f.patternKeys && f.patternKeys.length ? f.patternKeys : (f.patternKey ? [f.patternKey] : ['?']));
                    var tf = (f.patternTf||'d').toUpperCase();
                    var tipText = pKeys.map(function(k){ return AL_HIST_PAT_LABELS[k] || k.replace(/_/g,' '); }).join('\n');
                    if (pKeys.length === 1) {
                        return '<span class="al-hist-cond al-pat-single" style="color:#a78bfa;" data-al-tip="' + tipText + '">' + (AL_HIST_PAT_LABELS[pKeys[0]] || pKeys[0].replace(/_/g,' ')) + ' <span class="al-pat-single-tf">' + tf + '</span></span>';
                    }
                    return '<span class="al-hist-cond al-pat-multi" data-al-tip="' + tipText + '"><svg width="16" height="12" viewBox="0 0 16 12" fill="none" style="flex-shrink:0;vertical-align:middle;margin-right:2px"><polyline points="0,9 3,9 5,3 7,10 9,6 11,7 13,4 16,4" stroke="#a78bfa" stroke-width="1.5" fill="none" stroke-linejoin="round" stroke-linecap="round"/></svg><span class="al-pat-multi-count">' + pKeys.length + '</span><span class="al-pat-multi-tf">' + tf + '</span></span>';
                })()
                : f.condition === 'above'
                    ? '<span class="al-hist-cond" style="color:#3fb950;">▲ $' + (f.alertPrice||0).toFixed(2) + '</span>'
                    : '<span class="al-hist-cond" style="color:#f85149;">▼ $' + (f.alertPrice||0).toFixed(2) + '</span>';
            var hitCls = f.condition === 'above' ? 'up' : 'dn';
            var name = esc(f.name || '');
            var curPrice = alertPrices[f.ticker];
            var chgHtml = '';
            if (curPrice != null && f.alertPrice > 0 && f.alertType !== 'ma') {
                var chgPct = ((curPrice - f.alertPrice) / f.alertPrice) * 100;
                var chgCls = chgPct > 0.05 ? 'up' : chgPct < -0.05 ? 'dn' : 'flat';
                var chgSign = chgPct > 0 ? '+' : '';
                chgHtml = '<div class="al-hist-row-bot">' +
                    '<span class="al-hist-since-label">since alert</span>' +
                    '<span class="al-hist-chg ' + chgCls + '">' + chgSign + chgPct.toFixed(2) + '%</span>' +
                '</div>';
            }
            return '<div class="al-hist-row">' +
                '<div class="al-hist-row-top">' +
                    '<span class="al-hist-ticker" onclick="alTickerClick(\'' + esc(f.ticker) + '\')" style="cursor:pointer;">' + esc(f.ticker) + '</span>' +
                    '<span class="al-hist-time">' + ts + '</span>' +
                '</div>' +
                (name ? '<div class="al-hist-name">' + name + '</div>' : '') +
                '<div class="al-hist-row-mid">' +
                    condHtml +
                    '<span class="al-hist-hit ' + hitCls + '">' + (curPrice != null ? 'now $' + curPrice.toFixed(2) : 'now —') + '</span>' +
                '</div>' +
                chgHtml +
                '<button class="al-hist-del" onclick="alHistDelete(' + idx + ')">×</button>' +
            '</div>';
        }).join('');
    }

    window.alHistOpen = function() {
        document.getElementById('al-hist-panel').classList.add('open');
        document.getElementById('al-hist-tab').style.display = 'none';
        var exp = document.getElementById('al-hist-expanded');
        exp.classList.add('open');
        exp.style.display = 'flex';
        renderHistory();
        // Fetch latest prices for history tickers then re-render with real values
        if (alertFiredList.length) {
            alFetchPrices().then(function() { renderHistory(); }).catch(function() {});
        }
    };

    window.alHistClose = function() {
        document.getElementById('al-hist-panel').classList.remove('open');
        document.getElementById('al-hist-tab').style.display = 'flex';
        var exp = document.getElementById('al-hist-expanded');
        exp.classList.remove('open');
        exp.style.display = 'none';
    };

    window.alListOpen = function() {
        var panel = document.getElementById('al-list-panel');
        panel.classList.add('open');
        document.getElementById('al-list-tab').style.display = 'none';
        var exp = document.getElementById('al-list-expanded');
        exp.classList.add('open');
        exp.style.display = 'flex';
    };

    window.alListClose = function() {
        var panel = document.getElementById('al-list-panel');
        panel.classList.remove('open');
        document.getElementById('al-list-tab').style.display = 'flex';
        var exp = document.getElementById('al-list-expanded');
        exp.classList.remove('open');
        exp.style.display = 'none';
    };

    window.alHistDelete = function(idx) {
        alertFiredList.splice(idx, 1);
        alSaveFired();
        alUpdateBadge();
        renderHistory();
    };

    window.alHistClearAll = function() {
        alertFiredList = [];
        alSaveFired();
        alUpdateBadge();
        renderHistory();
    };

    window.alFormTypeChange = function() {
        var type     = document.getElementById('al-input-type').value;
        var cond     = document.getElementById('al-input-cond');
        var price    = document.getElementById('al-input-price');
        var note     = document.getElementById('al-rsi-note');
        var rowMA    = document.getElementById('al-row-ma');
        var rowMA1   = document.getElementById('al-row-ma1');
        var rowMA2   = document.getElementById('al-row-ma2');
        var rowValue = document.getElementById('al-row-value');
        var rowCond  = document.getElementById('al-row-cond');
        var rowPat   = document.getElementById('al-row-pattern');
        var rowPatTf = document.getElementById('al-row-pattern-tf');
        if (type === 'rsi14') {
            cond.innerHTML = '<option value="above">above</option><option value="below">below</option>';
            price.placeholder = 'RSI'; price.step = '1'; price.min = '1'; price.max = '99'; price.value = '';
            if (note) note.style.display = '';
            rowMA.style.display = 'none'; rowMA1.style.display = 'none'; rowMA2.style.display = 'none';
            rowValue.style.display = '';
            if (rowCond)  rowCond.style.display  = '';
            if (rowPat)   rowPat.style.display   = 'none';
            if (rowPatTf) rowPatTf.style.display = 'none';
        } else if (type === 'ma') {
            cond.innerHTML =
                '<option value="price_above">price crosses above</option>' +
                '<option value="price_below">price crosses below</option>' +
                '<option value="ma1_above">MA crosses above MA</option>' +
                '<option value="ma1_below">MA crosses below MA</option>';
            if (note) note.style.display = 'none';
            rowValue.style.display = 'none';
            if (rowCond)  rowCond.style.display  = '';
            if (rowPat)   rowPat.style.display   = 'none';
            if (rowPatTf) rowPatTf.style.display = 'none';
            alMACondChange();
        } else if (type === 'pattern') {
            if (note) note.style.display = 'none';
            rowMA.style.display = 'none'; rowMA1.style.display = 'none'; rowMA2.style.display = 'none';
            rowValue.style.display = 'none';
            if (rowCond)  rowCond.style.display  = 'none';
            if (rowPat)   rowPat.style.display   = '';
            if (rowPatTf) rowPatTf.style.display = '';
        } else {
            cond.innerHTML = '<option value="above">crosses above</option><option value="below">crosses below</option><option value="prevdayhigh">prev day high</option><option value="prevdaylow">prev day low</option>';
            price.placeholder = 'Price'; price.step = '0.01'; price.min = '0'; price.removeAttribute('max'); price.value = '';
            if (note) note.style.display = 'none';
            rowMA.style.display = 'none'; rowMA1.style.display = 'none'; rowMA2.style.display = 'none';
            if (rowCond)  rowCond.style.display  = '';
            if (rowPat)   rowPat.style.display   = 'none';
            if (rowPatTf) rowPatTf.style.display = 'none';
            alCondChange();
        }
    };

    window.alMACondChange = function() {
        var type = document.getElementById('al-input-type').value;
        if (type !== 'ma') return;
        var condVal  = document.getElementById('al-input-cond').value;
        var isMAvsMA = condVal === 'ma1_above' || condVal === 'ma1_below';
        document.getElementById('al-row-ma').style.display  = isMAvsMA ? 'none' : '';
        document.getElementById('al-row-ma1').style.display = isMAvsMA ? '' : 'none';
        document.getElementById('al-row-ma2').style.display = isMAvsMA ? '' : 'none';
    };

    window.alCondChange = function() {
        var type    = document.getElementById('al-input-type').value;
        var condVal = document.getElementById('al-input-cond').value;
        var rowValue  = document.getElementById('al-row-value');
        var rowCandle = document.getElementById('al-row-candle');
        if (type === 'ma') {
            alMACondChange();
        } else if (type === 'price') {
            var isPrevDay = condVal === 'prevdayhigh' || condVal === 'prevdaylow';
            if (rowValue)  rowValue.style.display  = isPrevDay ? 'none' : '';
            if (rowCandle) rowCandle.style.display = isPrevDay ? '' : 'none';
            if (!isPrevDay) alCandleSelect(1);
            var btn52 = document.getElementById('al-candle-52wk');
            if (btn52) btn52.textContent = condVal === 'prevdaylow' ? '52W L' : '52W H';
        }
    };

    window._alCandleOffset = 1;
    window.alCandleSelect = function(n) {
        window._alCandleOffset = n;
        [1, 2, 3, 4].forEach(function(i) {
            var btn = document.getElementById('al-candle-' + i);
            if (btn) btn.classList.toggle('active', i === n);
        });
        var btn52 = document.getElementById('al-candle-52wk');
        if (btn52) btn52.classList.toggle('active', n === '52wk');
    };

    window._alPatternTf = 'd';
    window.alPatternTfSelect = function(tf) {
        window._alPatternTf = tf;
        ['d','w','m'].forEach(function(t) {
            var btn = document.getElementById('al-ptf-' + t);
            if (btn) btn.classList.toggle('active', t === tf);
        });
    };

    // Multi-pattern helpers
    window.alPatChipToggle = function(chip) {
        var cb = chip.querySelector('input[type=checkbox]');
        cb.checked = !cb.checked;
        chip.classList.toggle('selected', cb.checked);
    };

    window.alGetSelectedPatterns = function() {
        var grid = document.getElementById('al-pattern-grid');
        if (!grid) return ['inside_day'];
        var checked = grid.querySelectorAll('input[type=checkbox]:checked');
        var keys = [];
        checked.forEach(function(cb) { keys.push(cb.value); });
        return keys;
    };

    window.alSetSelectedPatterns = function(keys) {
        var grid = document.getElementById('al-pattern-grid');
        if (!grid) return;
        var keySet = {};
        (keys || ['inside_day']).forEach(function(k) { keySet[k] = true; });
        grid.querySelectorAll('label.al-pat-chip').forEach(function(chip) {
            var cb = chip.querySelector('input[type=checkbox]');
            var on = !!keySet[cb.value];
            cb.checked = on;
            chip.classList.toggle('selected', on);
        });
    };

    // Returns array of pattern keys for an alert — handles old single-key & new multi-key
    window.alGetPatternKeys = function(a) {
        if (a.patternKeys && a.patternKeys.length) return a.patternKeys;
        if (a.patternKey) return [a.patternKey];
        return ['inside_day'];
    };

    // Canonical dedup key for a pattern alert
    window.alPatternAlertKey = function(a) {
        var keys = window.alGetPatternKeys(a).slice().sort();
        return a.ticker + '_pattern_' + keys.join('+') + '_' + (a.patternTf || 'd');
    };

    window.alShowForm = function(prefillTicker) {
        if (window.Notification && Notification.permission === 'default') Notification.requestPermission();
        _alEditIdx = null;
        var tickerInput = document.getElementById('al-input-ticker');
        tickerInput.readOnly = false;
        tickerInput.value = prefillTicker || '';
        document.getElementById('al-input-price').value = '';
        document.getElementById('al-input-type').value = 'price';
        alFormTypeChange();
        document.getElementById('al-input-ma').value  = 'SMA50';
        document.getElementById('al-input-ma1').value = 'SMA5';
        document.getElementById('al-input-ma2').value = 'SMA50';
        alSetSelectedPatterns([]);
        alPatternTfSelect('d');
        document.getElementById('al-modal-confirm-btn').textContent = 'Set alert';
        document.getElementById('al-modal-title').textContent = 'Add Alert';
        document.getElementById('al-modal-overlay').classList.add('open');
        setTimeout(function() {
            if (prefillTicker) { document.getElementById('al-input-price').focus(); }
            else { tickerInput.focus(); }
        }, 50);
    };

    window.alEditOpen = function(idx) {
        var a = alertsList[idx];
        if (!a) return;
        _alEditIdx = idx;
        document.getElementById('al-input-ticker').value = a.ticker;
        document.getElementById('al-input-ticker').readOnly = true;
        var uiType = (a.alertType === 'rsi14') ? 'rsi14' : (a.alertType === 'ma' || a.alertType === 'macross') ? 'ma' : (a.alertType === 'pattern') ? 'pattern' : 'price';
        document.getElementById('al-input-type').value = uiType;
        alFormTypeChange();
        if (a.alertType === 'macross') {
            document.getElementById('al-input-cond').value = a.condition === 'above' ? 'ma1_above' : 'ma1_below';
            alMACondChange();
            document.getElementById('al-input-ma1').value = a.ma1Key || 'SMA5';
            document.getElementById('al-input-ma2').value = a.ma2Key || 'SMA50';
        } else if (a.alertType === 'ma') {
            document.getElementById('al-input-cond').value = a.condition === 'above' ? 'price_above' : 'price_below';
            alMACondChange();
            document.getElementById('al-input-ma').value = a.maKey || 'SMA50';
        } else if (a.alertType === 'pattern') {
            alSetSelectedPatterns(window.alGetPatternKeys(a));
            alPatternTfSelect(a.patternTf || 'd');
        } else {
            if (a.isPrevDay) {
                document.getElementById('al-input-cond').value = (a.isPrevDay === 'high' || a.isPrevDay === '52wk-high') ? 'prevdayhigh' : 'prevdaylow';
                alCondChange();
                alCandleSelect(a.isPrevDay === '52wk-high' || a.isPrevDay === '52wk-low' ? '52wk' : (a.prevDayCandle || 1));
            } else {
                document.getElementById('al-input-cond').value = a.condition;
                document.getElementById('al-input-price').value = a.price;
            }
        }
        document.getElementById('al-modal-confirm-btn').textContent = 'Update';
        document.getElementById('al-modal-title').textContent = 'Edit Alert';
        document.getElementById('al-modal-overlay').classList.add('open');
        setTimeout(function() {
            if (a.alertType === 'macross') document.getElementById('al-input-ma1').focus();
            else if (a.alertType === 'ma') document.getElementById('al-input-ma').focus();
            else if (a.alertType === 'pattern') { /* chips — no text focus needed */ }
            else document.getElementById('al-input-price').focus();
        }, 50);
    };

    window.alHideForm = function() {
        document.getElementById('al-modal-overlay').classList.remove('open');
        document.getElementById('al-input-ticker').value = '';
        document.getElementById('al-input-ticker').readOnly = false;
        document.getElementById('al-input-price').value = '';
        document.getElementById('al-input-type').value = 'price';
        document.getElementById('al-input-ma').value  = 'SMA50';
        document.getElementById('al-input-ma1').value = 'SMA5';
        document.getElementById('al-input-ma2').value = 'SMA50';
        alSetSelectedPatterns([]);
        alPatternTfSelect('d');
        alCandleSelect(1);
        alFormTypeChange();
        document.getElementById('al-modal-confirm-btn').textContent = 'Set alert';
        document.getElementById('al-modal-confirm-btn').disabled = false;
        _alEditIdx = null;
    };

    window.alSubmitForm = function() {
        var ticker    = document.getElementById('al-input-ticker').value.trim().toUpperCase();
        var uiType    = document.getElementById('al-input-type').value;
        var uiCond    = document.getElementById('al-input-cond').value;
        var price     = parseFloat(document.getElementById('al-input-price').value);
        var maKey     = document.getElementById('al-input-ma').value;
        var ma1Key    = document.getElementById('al-input-ma1').value;
        var ma2Key    = document.getElementById('al-input-ma2').value;
        var patternKeys = alGetSelectedPatterns();
        var patternTf  = window._alPatternTf || 'd';

        // Resolve alertType and condition from UI values
        var alertType, cond;
        if (uiType === 'ma') {
            if (uiCond === 'ma1_above') { alertType = 'macross'; cond = 'above'; }
            else if (uiCond === 'ma1_below') { alertType = 'macross'; cond = 'below'; }
            else if (uiCond === 'price_above') { alertType = 'ma'; cond = 'above'; }
            else { alertType = 'ma'; cond = 'below'; }
        } else if (uiType === 'rsi14') {
            alertType = 'rsi14'; cond = uiCond;
        } else if (uiType === 'pattern') {
            alertType = 'pattern'; cond = 'detected';
        } else {
            alertType = 'price'; cond = uiCond;
        }

        if (!ticker) { document.getElementById('al-input-ticker').focus(); return; }
        if (alertType === 'pattern' && !patternKeys.length) {
            var grid = document.getElementById('al-pattern-grid');
            if (grid) grid.style.outline = '1px solid #f85149';
            setTimeout(function(){ if (grid) grid.style.outline = ''; }, 1200);
            return;
        }
        if (alertType !== 'ma' && alertType !== 'macross' && alertType !== 'pattern') {
            if (uiCond !== 'prevdayhigh' && uiCond !== 'prevdaylow') {
                if (!price || price <= 0) { document.getElementById('al-input-price').focus(); return; }
                if (alertType === 'rsi14' && (price < 1 || price > 99)) { document.getElementById('al-input-price').focus(); return; }
            }
        }
        if (alertType === 'macross' && ma1Key === ma2Key) {
            document.getElementById('al-input-ma2').focus(); return;
        }

        var alKey = function(a) {
            if (a.alertType === 'macross') return a.ticker + '_macross_' + a.ma1Key + '_' + a.ma2Key + '_' + a.condition;
            if (a.alertType === 'ma') return a.ticker + '_ma_' + a.maKey + '_' + a.condition;
            if (a.alertType === 'pattern') return window.alPatternAlertKey(a);
            return a.ticker + '_' + a.price + '_' + a.condition;
        };

        if (_alEditIdx !== null) {
            // ── Prev Day High / Low edit: re-fetch ──
            if (uiCond === 'prevdayhigh' || uiCond === 'prevdaylow') {
                var isPrevDayHighE = uiCond === 'prevdayhigh';
                var candleOffsetE  = window._alCandleOffset || 1;
                var is52wkE        = candleOffsetE === '52wk';
                var confirmBtnE = document.getElementById('al-modal-confirm-btn');
                confirmBtnE.textContent = 'Fetching…';
                confirmBtnE.disabled = true;
                var editIdxCapture = _alEditIdx;
                fetch(WL_PROXY + '?symbol=' + encodeURIComponent(ticker) + '&interval=1d&range=5d')
                    .then(function(r) { return r.ok ? r.json() : null; })
                    .then(function(data) {
                        var result = data && data.chart && data.chart.result && data.chart.result[0];
                        var quote  = result && result.indicators && result.indicators.quote && result.indicators.quote[0];
                        var pdPrice;
                        if (is52wkE) {
                            var meta52E = result && result.meta;
                            pdPrice = meta52E && (isPrevDayHighE ? meta52E.fiftyTwoWeekHigh : meta52E.fiftyTwoWeekLow);
                        } else {
                            var len = quote && quote.high && quote.high.length;
                            pdPrice = len && (isPrevDayHighE ? quote.high[len - candleOffsetE] : quote.low[len - candleOffsetE]);
                        }
                        if (!pdPrice || pdPrice <= 0) {
                            confirmBtnE.textContent = 'Update';
                            confirmBtnE.disabled = false;
                            return;
                        }
                        pdPrice = parseFloat(pdPrice.toFixed(2));
                        var ae = alertsList[editIdxCapture];
                        if (ae) {
                            delete _alertFiredSess[alKey(ae)];
                            ae.condition     = isPrevDayHighE ? 'above' : 'below';
                            ae.alertType     = 'price';
                            ae.price         = pdPrice;
                            ae.isPrevDay     = is52wkE ? (isPrevDayHighE ? '52wk-high' : '52wk-low') : (isPrevDayHighE ? 'high' : 'low');
                            ae.prevDayCandle = is52wkE ? null : candleOffsetE;
                            delete ae.maKey; delete ae.ma1Key; delete ae.ma2Key;
                            delete ae.patternKey; delete ae.patternKeys; delete ae.patternTf;
                        }
                        alSave();
                        alHideForm();
                        if (!alertPriceTimer && !alertOpenTimer) alStartBackgroundPolling();
                        renderAlerts();
                    })
                    .catch(function() {
                        confirmBtnE.textContent = 'Update';
                        confirmBtnE.disabled = false;
                    });
                return;
            }

            var a = alertsList[_alEditIdx];
            if (a) {
                delete _alertFiredSess[alKey(a)];
                a.condition = cond;
                a.alertType = alertType;
                if (alertType === 'macross') { a.ma1Key = ma1Key; a.ma2Key = ma2Key; delete a.maKey; delete a.patternKey; delete a.patternKeys; delete a.patternTf; a.price = 0; }
                else if (alertType === 'ma') { a.maKey = maKey; delete a.ma1Key; delete a.ma2Key; delete a.patternKey; delete a.patternKeys; delete a.patternTf; a.price = 0; }
                else if (alertType === 'pattern') { a.patternKeys = patternKeys; delete a.patternKey; a.patternTf = patternTf; delete a.maKey; delete a.ma1Key; delete a.ma2Key; a.price = 0; }
                else { a.price = price; delete a.maKey; delete a.ma1Key; delete a.ma2Key; delete a.patternKey; delete a.patternKeys; delete a.patternTf; }
            }
            alSave();
            alHideForm();
            if (!alertPriceTimer && !alertOpenTimer) alStartBackgroundPolling();
            renderAlerts();
        } else {
            // ── Prev Day High / Low: async fetch then save ──
            if (uiCond === 'prevdayhigh' || uiCond === 'prevdaylow') {
                var isPrevDayHigh = uiCond === 'prevdayhigh';
                var candleOffset  = window._alCandleOffset || 1;
                var is52wk        = candleOffset === '52wk';
                var confirmBtn = document.getElementById('al-modal-confirm-btn');
                confirmBtn.textContent = 'Fetching…';
                confirmBtn.disabled = true;
                fetch(WL_PROXY + '?symbol=' + encodeURIComponent(ticker) + '&interval=1d&range=5d')
                    .then(function(r) { return r.ok ? r.json() : null; })
                    .then(function(data) {
                        var result = data && data.chart && data.chart.result && data.chart.result[0];
                        var quote  = result && result.indicators && result.indicators.quote && result.indicators.quote[0];
                        var pdPrice;
                        if (is52wk) {
                            var meta52 = result && result.meta;
                            pdPrice = meta52 && (isPrevDayHigh ? meta52.fiftyTwoWeekHigh : meta52.fiftyTwoWeekLow);
                        } else {
                            var len = quote && quote.high && quote.high.length;
                            pdPrice = len && (isPrevDayHigh ? quote.high[len - candleOffset] : quote.low[len - candleOffset]);
                        }
                        if (!pdPrice || pdPrice <= 0) {
                            confirmBtn.textContent = 'Set alert';
                            confirmBtn.disabled = false;
                            return;
                        }
                        pdPrice = parseFloat(pdPrice.toFixed(2));
                        var meta = result && result.meta;
                        var resolvedName = (meta && (meta.shortName || meta.longName)) || '';
                        var pdEntry = {
                            ticker: ticker,
                            condition: isPrevDayHigh ? 'above' : 'below',
                            price: pdPrice,
                            alertType: 'price',
                            isPrevDay: is52wk ? (isPrevDayHigh ? '52wk-high' : '52wk-low') : (isPrevDayHigh ? 'high' : 'low'),
                            prevDayCandle: is52wk ? null : candleOffset,
                            name: resolvedName,
                            addedAt: new Date().toISOString()
                        };
                        alertsList.push(pdEntry);
                        delete _alertFiredSess[alKey(pdEntry)];
                        alSave();
                        alHideForm();
                        alStartBackgroundPolling();
                        renderAlerts();
                        var _alList = document.getElementById('al-list');
                        if (_alList) _alList.scrollTop = _alList.scrollHeight;
                    })
                    .catch(function() {
                        confirmBtn.textContent = 'Set alert';
                        confirmBtn.disabled = false;
                    });
                return;
            }

            var entry;
            if (alertType === 'macross') {
                entry = { ticker: ticker, condition: cond, price: 0, alertType: 'macross', ma1Key: ma1Key, ma2Key: ma2Key, name: '', addedAt: new Date().toISOString() };
            } else if (alertType === 'ma') {
                entry = { ticker: ticker, condition: cond, price: 0, alertType: 'ma', maKey: maKey, name: '', addedAt: new Date().toISOString() };
            } else if (alertType === 'pattern') {
                entry = { ticker: ticker, condition: 'detected', price: 0, alertType: 'pattern', patternKeys: patternKeys, patternTf: patternTf, name: '', addedAt: new Date().toISOString() };
            } else {
                entry = { ticker: ticker, condition: cond, price: price, alertType: alertType, name: '', addedAt: new Date().toISOString() };
            }
            var entryIdx = alertsList.length;
            alertsList.push(entry);
            delete _alertFiredSess[alKey(entry)];
            alSave();
            alHideForm();
            alStartBackgroundPolling();
            renderAlerts();
            var _alList = document.getElementById('al-list');
            if (_alList) _alList.scrollTop = _alList.scrollHeight;
            (function(capturedTicker, capturedIdx) {
                fetch(WL_PROXY + '?symbol=' + encodeURIComponent(capturedTicker) + '&interval=1d&range=2d')
                    .then(function(r) { return r.ok ? r.json() : null; })
                    .then(function(data) {
                        var meta = data && data.chart && data.chart.result && data.chart.result[0] && data.chart.result[0].meta;
                        var resolvedName = (meta && (meta.shortName || meta.longName)) || '';
                        var target = alertsList[capturedIdx];
                        if (resolvedName && target && target.ticker === capturedTicker && !target.name) {
                            target.name = resolvedName;
                            alSave();
                            renderAlerts();
                        }
                    }).catch(function() {});
            })(ticker, entryIdx);
        }
    };

    window.alDelete = function(idx) {
        var a = alertsList[idx];
        if (a) {
            var k;
            if (a.alertType === 'macross') k = a.ticker + '_macross_' + a.ma1Key + '_' + a.ma2Key + '_' + a.condition;
            else if (a.alertType === 'ma') k = a.ticker + '_ma_' + a.maKey + '_' + a.condition;
            else if (a.alertType === 'pattern') k = window.alPatternAlertKey(a);
            else k = a.ticker + '_' + a.price + '_' + a.condition;
            delete _alertFiredSess[k];
        }
        alertsList.splice(idx, 1);
        alSave();
        alUpdateBadge();
        if (!alertsList.length) {
            if (alertPriceTimer) { clearInterval(alertPriceTimer); alertPriceTimer = null; }
            if (alertOpenTimer)  { clearTimeout(alertOpenTimer);   alertOpenTimer  = null; }
        }
        renderAlerts();
    };

    window.alDismissMissed = function() {
        alertFiredList.forEach(function(f) { f.dismissed = true; });
        alSaveFired();
        alUpdateBadge();
        renderAlerts();
    };

    window.alOpenChart = function(ticker) {
        var sd = tickerMap && tickerMap[ticker];
        openChartModal(ticker);
    };

    // ── Alerts inline chart panel (LW Charts) ────────────────────────────
    // State — mirrors _wl* for the watchlist side-panel chart
    var _alOhlcv              = [];
    var _alSym                = null;
    var _alChartTf            = 'D';
    var _alLastCrosshairPrice = null;
    var _alChart              = null;
    var _alCandle             = null;
    var _alVol                = null;
    var _alVolMa              = null;
    var _alVolData            = null;
    var _alMaSeries           = {};
    var _alMaDataMap          = {};
    var _alLastCrosshairTime  = null;
    var _alVwapSeries         = [];
    var _alVwapMode           = false;
    var _alVisibleBars        = 252;
    var _alActiveMas          = { SMA5: true, EMA8: true, EMA21: true, SMA50: true, SMA150: true, SMA200: true };
    var _alKeyHandler         = null;
    var _alTrendlineMode      = false;
    var _alTrendlines         = [];
    var _alTrendlineFirst     = null;
    var _alTrendSvgOverlay    = null;
    var _alTrendSvgLine       = null;
    var _alTrendDraw          = { active: false, startTime: null, startPrice: null };
    var _alTrendContRef       = null;
    var _alTrendMoveBound     = false;
    var _alSelectedTrendlineIdx = -1;
    var _alSelectedVwapIdx      = -1;
    var _alTrendDragState       = null;
    var _alCtxPrice             = null;
    var _alCtxMa                = null;
    var _alCtxAttached          = false;

    // Measure tool state (al)
    var _alMeasureMode       = false;
    var _alMeasureActive     = false;
    var _alMeasurePhase      = 0;
    var _alMeasureRafId      = null;
    var _alMeasureStart      = null;
    var _alMeasureResult     = null;
    var _alMeasureSvgOverlay = null;
    var _alMeasureSvgRect    = null;
    var _alMeasureHLine      = null;
    var _alMeasureInfoDiv    = null;

    var _alClickTimer   = null;
    var _alClickTicker  = null;

    // ── Click: single = side-panel, double = fullscreen ───────────────────
    window.alTickerClick = function(ticker) {
        if (_alClickTimer && _alClickTicker === ticker) {
            clearTimeout(_alClickTimer);
            _alClickTimer  = null;
            _alClickTicker = null;
            alOpenChart(ticker);
        } else {
            if (_alClickTimer) clearTimeout(_alClickTimer);
            _alClickTicker = ticker;
            _alClickTimer  = setTimeout(function() {
                _alClickTimer  = null;
                _alClickTicker = null;
                alSelectChart(ticker);
            }, 220);
        }
    };

    // ── Trendline helpers ─────────────────────────────────────────────────
    function _addAlTrendline(p1, p2) {
        if (!_alChart || !_alCandle || !_alOhlcv.length) return;
        var refChart  = _alChart;
        var refSeries = _alCandle;
        var ohlcv     = _alOhlcv;
        var leftP  = p1.time <= p2.time ? p1 : p2;
        var rightP = p1.time <= p2.time ? p2 : p1;
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
                                if (tlObj.dragging) return;
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
                                    ctx.beginPath();
                                    ctx.moveTo(bx1, by1);
                                    ctx.lineTo(bx2, by2);
                                    ctx.strokeStyle = _TRENDLINE_COLOR;
                                    ctx.lineWidth   = 1.5 * rx;
                                    ctx.stroke();
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
        _alTrendlines.push(tlObj);
    }

    function _alTrendlineHitTest(clientX, clientY) {
        if (!_alChart || !_alCandle || !_alTrendlines.length || !_alTrendContRef) return -1;
        var rect     = _alTrendContRef.getBoundingClientRect();
        var px       = clientX - rect.left;
        var py       = clientY - rect.top;
        var HIT_PX   = 7;
        var bestIdx  = -1;
        var bestDist = HIT_PX;
        _alTrendlines.forEach(function(tl, idx) {
            var x1 = _mcFsTimeToX(_alChart, _alOhlcv, tl.leftP.time);
            var x2 = _mcFsTimeToX(_alChart, _alOhlcv, tl.rightP.time);
            var y1 = _alCandle.priceToCoordinate(tl.leftP.price);
            var y2 = _alCandle.priceToCoordinate(tl.rightP.price);
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

    function _alDeselectAllTrendlines() {
        _alTrendlines.forEach(function(tl) {
            if (tl.selected) { tl.selected = false; if (tl.requestUpdate) tl.requestUpdate(); }
        });
        _alSelectedTrendlineIdx = -1;
    }

    function _alSelectVwap(idx) {
        _alVwapSeries.forEach(function(entry, i) {
            entry.series.applyOptions({ lineWidth: i === idx ? 3 : 1.5 });
        });
        _alSelectedVwapIdx = idx;
    }
    function _alDeselectAllVwaps() {
        _alVwapSeries.forEach(function(entry) {
            entry.series.applyOptions({ lineWidth: 1.5 });
        });
        _alSelectedVwapIdx = -1;
    }

    function _alAnchorHitTest(clientX, clientY, tlIdx) {
        if (tlIdx < 0 || !_alTrendlines[tlIdx] || !_alChart || !_alCandle || !_alTrendContRef) return null;
        var tl   = _alTrendlines[tlIdx];
        var rect = _alTrendContRef.getBoundingClientRect();
        var px   = clientX - rect.left;
        var py   = clientY - rect.top;
        var HIT  = 10;
        var x1 = _mcFsTimeToX(_alChart, _alOhlcv, tl.leftP.time);
        var y1 = _alCandle.priceToCoordinate(tl.leftP.price);
        if (x1 != null && y1 != null && Math.hypot(px - x1, py - y1) <= HIT) return 'left';
        var x2 = _mcFsTimeToX(_alChart, _alOhlcv, tl.rightP.time);
        var y2 = _alCandle.priceToCoordinate(tl.rightP.price);
        if (x2 != null && y2 != null && Math.hypot(px - x2, py - y2) <= HIT) return 'right';
        return null;
    }

    // ── Anchor drag ───────────────────────────────────────────────────────
    function _onAlTrendAnchorDragMove(evt) {
        if (!_alTrendDragState || !_alChart || !_alCandle || !_alTrendContRef) return;
        var tl = _alTrendlines[_alTrendDragState.tlIdx];
        if (!tl) return;
        if (_alTrendContRef) _alTrendContRef.style.cursor = 'grabbing';
        var rect  = _alTrendContRef.getBoundingClientRect();
        var lx    = evt.clientX - rect.left;
        var ly    = evt.clientY - rect.top;
        var price = _alCandle.coordinateToPrice(ly);
        var time  = _alChart.timeScale().coordinateToTime(lx);
        if (price == null) return;
        if (time == null) {
            var ohlcv  = _alOhlcv;
            var last   = ohlcv[ohlcv.length - 1];
            var prev   = ohlcv[ohlcv.length - 2] || last;
            var barSec = ohlcv.length >= 2 ? (last.time - prev.time) : 86400;
            var lastX  = _alChart.timeScale().timeToCoordinate(last.time);
            if (lastX == null) return;
            var prevX    = _alChart.timeScale().timeToCoordinate(prev.time);
            var pxPerBar = prevX != null ? Math.abs(lastX - prevX) : 8;
            var barsAhead = pxPerBar > 0 ? Math.max(1, Math.round((lx - lastX) / pxPerBar)) : 1;
            time = last.time + barsAhead * barSec;
        }
        var newAnchor = { time: time, price: price };
        if (_alTrendDragState.anchorSide === 'left') {
            tl.leftP = newAnchor;
        } else {
            tl.rightP = newAnchor;
        }
        if (tl.leftP.time > tl.rightP.time) {
            var tmp = tl.leftP; tl.leftP = tl.rightP; tl.rightP = tmp;
            _alTrendDragState.anchorSide = _alTrendDragState.anchorSide === 'left' ? 'right' : 'left';
        }
        tl.p1 = tl.leftP; tl.p2 = tl.rightP;
        if (_alTrendSvgOverlay && _alTrendSvgLine && _alTrendDragState.fixedX != null) {
            _alTrendSvgLine.setAttribute('x2', lx);
            _alTrendSvgLine.setAttribute('y2', ly);
        }
    }

    function _onAlTrendAnchorDragEnd() {
        var state = _alTrendDragState;
        _alTrendDragState = null;
        document.removeEventListener('mousemove', _onAlTrendAnchorDragMove);
        document.removeEventListener('mouseup',   _onAlTrendAnchorDragEnd);
        if (_alTrendContRef) _alTrendContRef.style.cursor = '';
        if (state) {
            var tl = _alTrendlines[state.tlIdx];
            if (tl) { tl.dragging = false; if (tl.requestUpdate) tl.requestUpdate(); }
        }
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
            });
        });
    }

    // ── Trendline Alert creation API ──────────────────────────────────────
    window.alAddTrendlineAlert = function(ticker, p1, p2, condition) {
        // Convert LW time to unix seconds (handles number, 'YYYY-MM-DD' string, or {year,month,day})
        function toUnix(t) {
            if (typeof t === 'number') return t;
            if (typeof t === 'string') return Math.floor(new Date(t).getTime() / 1000);
            if (t && t.year != null) return Math.floor(Date.UTC(t.year, t.month - 1, t.day) / 1000);
            return 0;
        }
        var u1 = toUnix(p1.time), u2 = toUnix(p2.time);
        // Normalise so np1.unix <= np2.unix (chronological order)
        var np1, np2;
        if (u1 <= u2) {
            np1 = { time: p1.time, price: p1.price, unix: u1 };
            np2 = { time: p2.time, price: p2.price, unix: u2 };
        } else {
            np1 = { time: p2.time, price: p2.price, unix: u2 };
            np2 = { time: p1.time, price: p1.price, unix: u1 };
        }
        // Dedup: same ticker + same two anchor times + same condition
        var exists = alertsList.some(function(a) {
            return a.alertType === 'trendline' && a.ticker === ticker &&
                   a.condition === condition && a.p1 && a.p2 &&
                   a.p1.unix === np1.unix && a.p2.unix === np2.unix;
        });
        if (exists) return;
        var name = (tickerMap && tickerMap[ticker] && tickerMap[ticker].name) ? tickerMap[ticker].name : '';
        alertsList.push({
            ticker:    ticker,
            alertType: 'trendline',
            condition: condition,
            p1:        np1,
            p2:        np2,
            name:      name,
            addedAt:   new Date().toISOString()
        });
        _alLoaded = true;
        alSave();
        alStartBackgroundPolling();
        if (currentView === 'alerts') renderAlerts();
        alStampBadges();
    };

    // Accessor for other chart contexts (fullscreen, watchlist) to read trendline alerts
    window.alGetTrendlineAlerts = function(ticker) {
        return alertsList.filter(function(a) {
            return a.alertType === 'trendline' && a.ticker === ticker && a.p1 && a.p2;
        });
    };

    // ── AL Measure drag handlers ─────────────────────────────────────────────
    function _onAlMeasureDragMove(evt) {
        if (!_alMeasureActive || !_alTrendContRef || !_alChart || !_alCandle) return;
        if (_alMeasureRafId) return;
        var cx = evt.clientX, cy = evt.clientY;
        _alMeasureRafId = requestAnimationFrame(function() {
            _alMeasureRafId = null;
            if (!_alMeasureActive) return;
            var r  = _alTrendContRef.getBoundingClientRect();
            var lx = cx - r.left;
            var ly = cy - r.top;
            var eP = _alCandle.coordinateToPrice(ly);
            var eT = _measureGetTimeAtX(_alChart, _alOhlcv, lx);
            if (eP == null || eT == null) return;
            _alMeasureResult = _computeMeasureResult(_alOhlcv, _alMeasureStart.time, _alMeasureStart.price, eT, eP);
            _renderMeasureOverlay(_alChart, _alCandle, _alTrendContRef,
                _alMeasureSvgOverlay, _alMeasureSvgRect, _alMeasureHLine,
                _alMeasureInfoDiv, _alMeasureResult);
        });
    }
    function _onAlMeasureDragEnd() {
        document.removeEventListener('mousemove', _onAlMeasureDragMove);
        document.removeEventListener('mouseup',   _onAlMeasureDragEnd);
        _alMeasureActive = false;
    }
    function _onAlMeasurePreviewMove(evt) {
        if (!_alMeasureActive || _alMeasurePhase !== 1 || !_alTrendContRef || !_alChart || !_alCandle) return;
        if (_alMeasureRafId) return;
        var cx = evt.clientX, cy = evt.clientY;
        _alMeasureRafId = requestAnimationFrame(function() {
            _alMeasureRafId = null;
            if (!_alMeasureActive || _alMeasurePhase !== 1) return;
            var r  = _alTrendContRef.getBoundingClientRect();
            var lx = cx - r.left;
            var ly = cy - r.top;
            var eP = _alCandle.coordinateToPrice(ly);
            var eT = _measureGetTimeAtX(_alChart, _alOhlcv, lx);
            if (eP == null || eT == null) return;
            _alMeasureResult = _computeMeasureResult(_alOhlcv, _alMeasureStart.time, _alMeasureStart.price, eT, eP);
            _renderMeasureOverlay(_alChart, _alCandle, _alTrendContRef,
                _alMeasureSvgOverlay, _alMeasureSvgRect, _alMeasureHLine,
                _alMeasureInfoDiv, _alMeasureResult);
        });
    }

    function _onAlTrendMouseDown(evt) {
        if (evt.button !== 0 || !_alCandle || !_alChart || !_alTrendContRef) return;

        // ── Measure tool intercept ────────────────────────────────────────────
        if ((evt.shiftKey || _alMeasureMode) && !_alTrendDragState) {
            evt.stopPropagation();
            evt.preventDefault();
            if (_alTrendDraw.active) {
                _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
                if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
            }
            var _mRect = _alTrendContRef.getBoundingClientRect();
            var _mlx   = evt.clientX - _mRect.left;
            var _mly   = evt.clientY - _mRect.top;
            var _mP    = _alCandle.coordinateToPrice(_mly);
            var _mT    = _measureGetTimeAtX(_alChart, _alOhlcv, _mlx);
            if (_mP == null || _mT == null) return;
            var _mSi   = _barIdxByTime(_alOhlcv, _mT);

            if (_alMeasurePhase === 1) {
                _alMeasureResult = _computeMeasureResult(_alOhlcv, _alMeasureStart.time, _alMeasureStart.price, _mT, _mP);
                _renderMeasureOverlay(_alChart, _alCandle, _alTrendContRef,
                    _alMeasureSvgOverlay, _alMeasureSvgRect, _alMeasureHLine,
                    _alMeasureInfoDiv, _alMeasureResult);
                _alMeasureActive = false;
                _alMeasurePhase  = 0;
                if (_alMeasureRafId) { cancelAnimationFrame(_alMeasureRafId); _alMeasureRafId = null; }
                document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
                return;
            }

            _alMeasureStart  = { time: _mT, price: _mP, barIdx: _mSi };
            _alMeasureResult = null;
            _alMeasureActive = true;
            _alMeasurePhase  = 1;
            _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
            document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
            document.addEventListener('mousemove', _onAlMeasurePreviewMove);
            return;
        }

        // Plain click (no shift, no measure mode) — cancel phase-1 preview or clear result
        if (_alMeasurePhase === 1) {
            _alMeasureActive = false;
            _alMeasurePhase  = 0;
            if (_alMeasureRafId) { cancelAnimationFrame(_alMeasureRafId); _alMeasureRafId = null; }
            document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
            _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
            _alMeasureResult = null;
        } else if (_alMeasureResult && !_alMeasureMode) {
            _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
            _alMeasureResult = null;
        }

        if (!_alTrendDraw.active) {
            var dragTlIdx = -1, anchorSide = null;
            if (_alSelectedTrendlineIdx !== -1) {
                anchorSide = _alAnchorHitTest(evt.clientX, evt.clientY, _alSelectedTrendlineIdx);
                if (anchorSide) dragTlIdx = _alSelectedTrendlineIdx;
            }
            if (dragTlIdx === -1) {
                for (var _di = 0; _di < _alTrendlines.length; _di++) {
                    var _as = _alAnchorHitTest(evt.clientX, evt.clientY, _di);
                    if (_as) { dragTlIdx = _di; anchorSide = _as; break; }
                }
            }
            if (dragTlIdx !== -1) {
                evt.stopPropagation();
                if (_alSelectedTrendlineIdx !== dragTlIdx) {
                    _alDeselectAllTrendlines();
                    _alSelectedTrendlineIdx = dragTlIdx;
                    _alTrendlines[dragTlIdx].selected = true;
                    if (_alTrendlines[dragTlIdx].requestUpdate) _alTrendlines[dragTlIdx].requestUpdate();
                }
                var _dragTl = _alTrendlines[dragTlIdx];
                var _fixedP = anchorSide === 'left' ? _dragTl.rightP : _dragTl.leftP;
                var _fixedX = _mcFsTimeToX(_alChart, _alOhlcv, _fixedP.time);
                var _fixedY = _alCandle.priceToCoordinate(_fixedP.price);
                _alTrendDragState = { tlIdx: dragTlIdx, anchorSide: anchorSide, fixedX: _fixedX, fixedY: _fixedY };
                _dragTl.dragging = true;
                if (_dragTl.requestUpdate) _dragTl.requestUpdate();
                if (_alTrendSvgOverlay && _alTrendSvgLine && _fixedX != null && _fixedY != null) {
                    var _dRect = _alTrendContRef.getBoundingClientRect();
                    var _curX  = evt.clientX - _dRect.left;
                    var _curY  = evt.clientY - _dRect.top;
                    _alTrendSvgLine.setAttribute('x1', _fixedX); _alTrendSvgLine.setAttribute('y1', _fixedY);
                    _alTrendSvgLine.setAttribute('x2', _curX);   _alTrendSvgLine.setAttribute('y2', _curY);
                    _alTrendSvgOverlay.style.display = '';
                }
                document.addEventListener('mousemove', _onAlTrendAnchorDragMove);
                document.addEventListener('mouseup',   _onAlTrendAnchorDragEnd);
                return;
            }
        }

        if (!_alTrendDraw.active) {
            var hitIdx = _alTrendlineHitTest(evt.clientX, evt.clientY);
            if (hitIdx !== -1) {
                evt.stopPropagation();
                if (_alSelectedTrendlineIdx !== -1 && _alSelectedTrendlineIdx !== hitIdx) {
                    var prev = _alTrendlines[_alSelectedTrendlineIdx];
                    if (prev) { prev.selected = false; if (prev.requestUpdate) prev.requestUpdate(); }
                }
                _alSelectedTrendlineIdx = hitIdx;
                _alTrendlines[hitIdx].selected = true;
                if (_alTrendlines[hitIdx].requestUpdate) _alTrendlines[hitIdx].requestUpdate();
                return;
            }
            if (_alSelectedTrendlineIdx !== -1) _alDeselectAllTrendlines();
        }

        if (!_alTrendlineMode) return;
        evt.stopPropagation();

        var rect  = _alTrendContRef.getBoundingClientRect();
        var lx    = evt.clientX - rect.left;
        var ly    = evt.clientY - rect.top;
        var price = _alCandle.coordinateToPrice(ly);
        var time  = null;
        if (_alOhlcv.length >= 2) {
            var _ohlcv   = _alOhlcv;
            var _last    = _ohlcv[_ohlcv.length - 1];
            var _prev    = _ohlcv[_ohlcv.length - 2];
            var _lastX   = _alChart.timeScale().timeToCoordinate(_last.time);
            var _prevX   = _alChart.timeScale().timeToCoordinate(_prev.time);
            var _pxPerBar = (_lastX != null && _prevX != null) ? Math.abs(_lastX - _prevX) : 8;
            if (_lastX != null && lx > _lastX + _pxPerBar * 0.5) {
                var _barSec    = _last.time - _prev.time;
                var _barsAhead = Math.max(1, Math.round((lx - _lastX) / _pxPerBar));
                time = _last.time + _barsAhead * _barSec;
            } else {
                time = _alLastCrosshairTime || _last.time;
            }
        }
        if (price == null || time == null) return;
        if (!_alTrendDraw.active) {
            _alTrendDraw.active     = true;
            _alTrendDraw.startTime  = time;
            _alTrendDraw.startPrice = price;
            if (_alTrendSvgOverlay && _alTrendSvgLine && _alChart) {
                var ax = _alChart.timeScale().timeToCoordinate(time);
                var ay = _alCandle.priceToCoordinate(price);
                if (ax != null && ay != null) {
                    _alTrendSvgLine.setAttribute('x1', ax); _alTrendSvgLine.setAttribute('y1', ay);
                    _alTrendSvgLine.setAttribute('x2', ax); _alTrendSvgLine.setAttribute('y2', ay);
                }
                _alTrendSvgOverlay.style.display = '';
            }
        } else {
            var p1 = { time: _alTrendDraw.startTime, price: _alTrendDraw.startPrice };
            _alTrendDraw.active = false;
            _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
            if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
            if (time !== p1.time) _addAlTrendline(p1, { time: time, price: price });
            _alTrendlineMode = false;
            var tDoneBtn = document.getElementById('al-chart-trendline-btn');
            if (tDoneBtn) tDoneBtn.classList.remove('active');
        }
    }

    function _onAlTrendMouseMove(evt) {
        if (_alTrendDraw.active) {
            if (!_alTrendSvgOverlay || !_alTrendSvgLine || !_alCandle || !_alChart || !_alTrendContRef) return;
            var rect  = _alTrendContRef.getBoundingClientRect();
            var curX  = evt.clientX - rect.left;
            var curY  = evt.clientY - rect.top;
            var startTime = _alTrendDraw.startTime;
            if (!startTime) return;
            var x1 = _alChart.timeScale().timeToCoordinate(startTime);
            var y1 = _alCandle.priceToCoordinate(_alTrendDraw.startPrice);
            if (x1 == null || y1 == null) return;
            _alTrendSvgLine.setAttribute('x1', x1);
            _alTrendSvgLine.setAttribute('y1', y1);
            _alTrendSvgLine.setAttribute('x2', curX);
            _alTrendSvgLine.setAttribute('y2', curY);
            return;
        }
        if (_alTrendlines.length && _alTrendContRef && !_alTrendlineMode) {
            if (_alTrendDragState) return;
            if (_alSelectedTrendlineIdx !== -1) {
                var anchorSide = _alAnchorHitTest(evt.clientX, evt.clientY, _alSelectedTrendlineIdx);
                if (anchorSide) { _alTrendContRef.style.cursor = 'grab'; return; }
            }
            var hitIdx = _alTrendlineHitTest(evt.clientX, evt.clientY);
            _alTrendContRef.style.cursor = hitIdx !== -1 ? 'pointer' : '';
        }
    }

    // ── Right-click context menu ──────────────────────────────────────────
    var _alCtxTrendline = null; // {p1, p2} when right-click lands on a trendline

    // Returns the unix timestamp to use when evaluating a trendline.
    // Daily/weekly/monthly bars have midnight-UTC anchors (divisible by 86400).
    // Using Date.now() directly shifts the evaluated trendline by up to 20 hours
    // relative to what the chart visually shows at today's bar, causing premature
    // triggers. Snap to today's midnight UTC so the evaluated price matches the
    // visual trendline at the current bar. Intraday anchors are not midnight-aligned
    // so they fall through to real wall-clock time, which is correct for them.
    function _alTlEvalUnix(p1unix, p2unix) {
        if (p1unix % 86400 === 0 && p2unix % 86400 === 0) {
            return Math.floor(Date.now() / 86400000) * 86400;
        }
        return Math.floor(Date.now() / 1000);
    }

    // Linear interpolation/extrapolation of trendline price at a given unix timestamp
    function _alTrendlinePriceAt(p1unix, p1price, p2unix, p2price, nowUnix) {
        if (p1unix === p2unix) return p1price;
        return p1price + (p2price - p1price) * (nowUnix - p1unix) / (p2unix - p1unix);
    }

    function _alDismissCtx() {
        document.getElementById('al-chart-ctx-menu').style.display = 'none';
        _alCtxPrice     = null;
        _alCtxMa        = null;
        _alCtxTrendline = null;
    }

    window.alChartCtxAlert = function(direction) {
        if (_alCtxTrendline) {
            var tl = _alCtxTrendline;
            _alDismissCtx();
            if (!_alSym) return;
            window.alAddTrendlineAlert(_alSym, tl.p1, tl.p2, direction);
            return;
        }
        if (_alCtxMa) {
            var maKey = _alCtxMa;
            _alDismissCtx();
            if (!_alSym) return;
            alShowForm(_alSym);
            setTimeout(function() {
                document.getElementById('al-input-type').value = 'ma';
                if (typeof alFormTypeChange === 'function') alFormTypeChange();
                document.getElementById('al-input-cond').value = direction === 'above' ? 'price_above' : 'price_below';
                if (typeof alMACondChange === 'function') alMACondChange();
                document.getElementById('al-input-ma').value = maKey;
                document.getElementById('al-input-ma').focus();
            }, 60);
        } else {
            var price = _alCtxPrice;
            _alDismissCtx();
            if (!_alSym || price == null) return;
            alShowForm(_alSym);
            setTimeout(function() {
                document.getElementById('al-input-type').value = 'price';
                if (typeof alFormTypeChange === 'function') alFormTypeChange();
                document.getElementById('al-input-cond').value = direction === 'above' ? 'above' : 'below';
                document.getElementById('al-input-price').value = price.toFixed(2);
                document.getElementById('al-input-price').focus();
            }, 60);
        }
    };

    function _alAttachCtxMenu() {
        if (_alCtxAttached) return;
        _alCtxAttached = true;
        var chartWrap = document.getElementById('al-chart-widget-wrap');
        var chartDiv  = document.getElementById('al-chart-widget');
        chartWrap.addEventListener('contextmenu', function(evt) {
            if (!chartDiv.contains(evt.target)) return;
            evt.preventDefault();
            evt.stopPropagation();
            // Right-click: cancel active measurement first (no context menu shown)
            if (_alMeasurePhase === 1) {
                _alMeasureActive = false;
                _alMeasurePhase  = 0;
                if (_alMeasureRafId) { cancelAnimationFrame(_alMeasureRafId); _alMeasureRafId = null; }
                document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
                _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
                _alMeasureResult = null;
                return;
            }
            if (_alMeasureResult) {
                _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
                _alMeasureResult = null;
                return;
            }
            if (_alTrendDraw.active) {
                _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
                if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
                return;
            }
            if (_alVwapMode) {
                _alVwapMode = false;
                var vBtn = document.getElementById('al-chart-vwap-btn');
                if (vBtn) vBtn.classList.remove('active');
                return;
            }
            if (!_alChart || !_alSym) return;
            // ── Trendline right-click: check hit before price/MA ──────────────
            var _tlHitIdx = _alTrendlineHitTest(evt.clientX, evt.clientY);
            if (_tlHitIdx !== -1) {
                var _tlHit = _alTrendlines[_tlHitIdx];
                _alCtxTrendline = { p1: _tlHit.p1, p2: _tlHit.p2 };
                _alCtxPrice = null;
                _alCtxMa    = null;
                document.getElementById('al-chart-ctx-label').textContent     = _alSym + ' · Trendline';
                document.getElementById('al-chart-ctx-above-txt').textContent = 'Alert above trendline';
                document.getElementById('al-chart-ctx-below-txt').textContent = 'Alert below trendline';
                var menu  = document.getElementById('al-chart-ctx-menu');
                menu.style.display = 'block';
                var mw = menu.offsetWidth  || 185;
                var mh = menu.offsetHeight || 90;
                var x  = Math.min(evt.clientX, window.innerWidth  - mw - 8);
                var y  = Math.min(evt.clientY, window.innerHeight - mh - 8);
                menu.style.left = x + 'px';
                menu.style.top  = y + 'px';
                setTimeout(function() {
                    function _tlDismiss(e) {
                        if (!menu.contains(e.target)) {
                            _alDismissCtx();
                            document.removeEventListener('mousedown', _tlDismiss, true);
                            document.removeEventListener('keydown',   _tlKd,      true);
                        }
                    }
                    function _tlKd(e) {
                        if (e.key === 'Escape') {
                            _alDismissCtx();
                            document.removeEventListener('mousedown', _tlDismiss, true);
                            document.removeEventListener('keydown',   _tlKd,      true);
                        }
                    }
                    document.addEventListener('mousedown', _tlDismiss, true);
                    document.addEventListener('keydown',   _tlKd,      true);
                }, 0);
                return;
            }
            var chartRect = chartDiv.getBoundingClientRect();
            var localY    = evt.clientY - chartRect.top;
            var price = _alLastCrosshairPrice;
            if (price == null || isNaN(price)) {
                if (_alCandle) {
                    var fallbackPrice = _alCandle.coordinateToPrice(localY);
                    if (fallbackPrice != null && !isNaN(fallbackPrice)) price = fallbackPrice;
                }
            }
            if (price == null || isNaN(price)) return;
            var nearestMa   = null;
            var nearestDist = 10;
            if (_alLastCrosshairTime) {
                Object.keys(_alMaDataMap).forEach(function(key) {
                    if (!_alMaSeries[key]) return;
                    var maVal = _alMaDataMap[key].get(_alLastCrosshairTime);
                    if (maVal == null) return;
                    var maCoord = _alMaSeries[key].priceToCoordinate(maVal);
                    if (maCoord == null) return;
                    var dist = Math.abs(localY - maCoord);
                    if (dist < nearestDist) { nearestDist = dist; nearestMa = key; }
                });
            }
            _alCtxPrice = price;
            _alCtxMa    = nearestMa;
            if (nearestMa) {
                var maLabel = _maLabel(nearestMa);
                document.getElementById('al-chart-ctx-label').textContent    = _alSym + ' · ' + maLabel;
                document.getElementById('al-chart-ctx-above-txt').textContent = 'Price crosses above ' + maLabel;
                document.getElementById('al-chart-ctx-below-txt').textContent = 'Price crosses below ' + maLabel;
            } else {
                var fmt = '$' + price.toFixed(2);
                document.getElementById('al-chart-ctx-label').textContent    = _alSym + ' · ' + fmt;
                document.getElementById('al-chart-ctx-above-txt').textContent = 'Alert above ' + fmt;
                document.getElementById('al-chart-ctx-below-txt').textContent = 'Alert below ' + fmt;
            }
            var menu  = document.getElementById('al-chart-ctx-menu');
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
                        _alDismissCtx();
                        document.removeEventListener('mousedown', _dismiss, true);
                        document.removeEventListener('keydown',   _kd,      true);
                    }
                }
                function _kd(e) {
                    if (e.key === 'Escape') {
                        _alDismissCtx();
                        document.removeEventListener('mousedown', _dismiss, true);
                        document.removeEventListener('keydown',   _kd,      true);
                    }
                }
                document.addEventListener('mousedown', _dismiss, true);
                document.addEventListener('keydown',   _kd,      true);
            }, 0);
        }, true);
    }

    // ── Core chart destroy / build ────────────────────────────────────────
    function _destroyAlChart() {
        if (_alChart) { try { _alChart.remove(); } catch(e) {} _alChart = null; }
        _alCandle = null; _alVol = null; _alVolMa = null; _alVolData = null;
        _alMaSeries = {}; _alMaDataMap = {};
        _alVwapSeries = []; _alTrendlines = []; _alTrendlineFirst = null;
        _alTrendSvgOverlay = null; _alTrendSvgLine = null;
        _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
        _alTrendlineMode = false; _alSelectedTrendlineIdx = -1; _alSelectedVwapIdx = -1;
        _alLastCrosshairTime = null; _alSym = null;
        if (_alKeyHandler) { document.removeEventListener('keydown', _alKeyHandler); _alKeyHandler = null; }
        var mktEl = document.getElementById('al-chart-mkt-info');
        if (mktEl) mktEl.style.display = 'none';
        var tBtn  = document.getElementById('al-chart-trendline-btn');
        if (tBtn)  tBtn.classList.remove('active');
        var vBtn  = document.getElementById('al-chart-vwap-btn');
        if (vBtn)  vBtn.classList.remove('active');
        var maPanel   = document.getElementById('al-chart-ma-panel');
        var maChevron = document.getElementById('al-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
    }

    function _buildAlChart(sym, ohlcv, tf) {
        var container = document.getElementById('al-chart-widget');
        container.innerHTML = '';
        _destroyAlChart();

        _alOhlcv = ohlcv;
        _alSym   = sym;
        _alChartTf = tf;
        _alLastCrosshairPrice = null;

        if (!window.LightweightCharts || !_alOhlcv.length) {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">No data</div>';
            return;
        }

        // SVG trendline overlay
        _alTrendContRef = container;
        container.removeEventListener('mousedown', _onAlTrendMouseDown, true);
        container.addEventListener('mousedown', _onAlTrendMouseDown, true);

        var _existingSvg = container.querySelector('.al-trend-svg-overlay');
        if (_existingSvg) {
            _alTrendSvgOverlay = _existingSvg;
            _alTrendSvgLine    = _existingSvg.querySelector('line');
        } else {
            _alTrendSvgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            _alTrendSvgOverlay.setAttribute('class', 'al-trend-svg-overlay');
            _alTrendSvgOverlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5;display:none;';
            _alTrendSvgLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            _alTrendSvgLine.setAttribute('stroke', _TRENDLINE_COLOR);
            _alTrendSvgLine.setAttribute('stroke-width', '1.5');
            _alTrendSvgLine.setAttribute('x1', '0'); _alTrendSvgLine.setAttribute('y1', '0');
            _alTrendSvgLine.setAttribute('x2', '0'); _alTrendSvgLine.setAttribute('y2', '0');
            _alTrendSvgOverlay.appendChild(_alTrendSvgLine);
            container.style.position = 'relative';
            container.appendChild(_alTrendSvgOverlay);
        }
        _alTrendSvgOverlay.style.display = 'none';

        // ── Measure tool overlay ───────────────────────────────────────────
        var _almOver = _ensureMeasureOverlay(container, 'al-measure-svg', 'al-measure-info');
        _alMeasureSvgOverlay = _almOver.svg;
        _alMeasureSvgRect    = _almOver.rect;
        _alMeasureHLine      = _almOver.hLine;
        _alMeasureInfoDiv    = _almOver.info;
        _alMeasureResult     = null;
        _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);

        container.removeEventListener('mousemove', _onAlTrendMouseMove);
        container.addEventListener('mousemove', _onAlTrendMouseMove);

        // Create LW chart
        _alChart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: { background: { color: '#0d1117' }, textColor: '#6e7681', panes: { separatorColor: '#161b22', separatorHoverColor: 'rgba(33,38,45,0.5)' } },
            grid:    { vertLines: { visible: false }, horzLines: { visible: false } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: '#21262d', textColor: '#6e7681', scaleMargins: { top: 0.05, bottom: 0.02 } },
            timeScale: { borderColor: '#21262d', timeVisible: false, secondsVisible: false, rightOffset: 12 },
            handleScroll: true, handleScale: true,
        });
        _alAttachCtxMenu();

        // Candle series
        _alCandle = _alChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: '#089981', downColor: '#b22833', borderVisible: false,
            wickUpColor: '#089981', wickDownColor: '#b22833',
            priceLineVisible: false, lastValueVisible: true,
        });
        _alCandle.setData(_alOhlcv);

        // Volume pane
        _alVol = _alChart.addSeries(LightweightCharts.HistogramSeries, {
            color: '#63a0f8', priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: true,
        }, 1);
        _alVol.setData(_alOhlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? 'rgba(24,72,204,0.5)' : 'rgba(248,81,73,0.35)' };
        }));
        _alVol.priceScale().applyOptions({ visible: true, borderColor: '#21262d', textColor: '#6e7681', minimumWidth: 60 });

        // 50 SMA on volume
        (function() {
            var period = 50;
            _alVolData = [];
            for (var i = period - 1; i < _alOhlcv.length; i++) {
                var sum = 0;
                for (var j = i - (period - 1); j <= i; j++) sum += (_alOhlcv[j].volume || 0);
                _alVolData.push({ time: _alOhlcv[i].time, value: sum / period });
            }
            _alVolMa = _alChart.addSeries(LightweightCharts.LineSeries, {
                color: '#1848cc', lineWidth: 1,
                priceLineVisible: false, lastValueVisible: true,
                crosshairMarkerVisible: false,
            }, 1);
            _alVolMa.setData(_alVolData);
        })();

        // Pin volume pane to ~22% height
        (function() {
            var panes = _alChart.panes();
            if (panes && panes.length >= 2) {
                var totalH = container ? container.offsetHeight : 700;
                panes[1].setHeight(Math.round(totalH * 0.22));
            }
        })();

        // Vol % vs 50-SMA label
        (function() {
            if (!_alVolData || !_alVolData.length || !_alOhlcv.length) return;
            var lastBar = _alOhlcv[_alOhlcv.length - 1];
            var lastVol = lastBar.volume;
            var sma50   = _alVolData[_alVolData.length - 1].value;
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
            lbl.id = 'al-chart-vol-pct-label';
            lbl.style.cssText = 'position:absolute;z-index:20;pointer-events:none;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:3px;white-space:nowrap;line-height:1;';
            lbl.innerHTML = '<span style="color:#484f58;">›</span>'
                          + '<span style="color:' + color + ';">' + sign + volDiffPct.toFixed(1) + '%</span>';
            container.appendChild(lbl);
            setTimeout(function() {
                if (!_alChart) return;
                var volPaneTop = 0, volPaneH = 0;
                try {
                    var panes = _alChart.panes();
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
                    if (!lbl.isConnected || !_alChart) return;
                    var lastX = _alChart.timeScale().timeToCoordinate(lastBar.time);
                    if (lastX == null || lastX < 0) { lbl.style.display = 'none'; return; }
                    lbl.style.display = 'flex';
                    lbl.style.left = (lastX + 10) + 'px';
                    lbl.style.top  = lblTop;
                }
                positionVolLabel();
                _alChart.timeScale().subscribeVisibleTimeRangeChange(positionVolLabel);
            }, 60);
        })();

        // Active MAs
        Object.keys(_alActiveMas).forEach(function(key) {
            if (!_alActiveMas[key]) return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = _alChart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
            var maData = _calcMA(_alOhlcv, key);
            s.setData(maData);
            _alMaSeries[key]  = s;
            _alMaDataMap[key] = new Map(maData.map(function(d) { return [d.time, d.value]; }));
        });

        // Visible range
        var n = _alOhlcv.length;
        _alChart.timeScale().setVisibleLogicalRange({ from: n - _alVisibleBars, to: n + 12 });

        // Re-render measure overlay on pan/zoom
        _alChart.timeScale().subscribeVisibleLogicalRangeChange(function() {
            if (_alMeasureResult) {
                _renderMeasureOverlay(_alChart, _alCandle, _alTrendContRef,
                    _alMeasureSvgOverlay, _alMeasureSvgRect, _alMeasureHLine,
                    _alMeasureInfoDiv, _alMeasureResult);
            }
        });

        // Click: AVWAP + selection
        _alChart.subscribeClick(function(param) {
            if (_alVwapMode) {
                if (!param.time) return;
                var idx = _barIdxByTime(_alOhlcv, param.time);
                if (idx < 0) return;
                var color = _AVWAP_COLOR;
                var data  = _calcAVWAP(_alOhlcv, idx);
                var dataMap = new Map(data.map(function(d) { return [d.time, d.value]; }));
                var s = _alChart.addSeries(LightweightCharts.LineSeries, { color: color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
                s.setData(data);
                _alVwapSeries.push({ series: s, anchor: idx, color: color, dataMap: dataMap });
                return;
            }
            if (_alTrendlineMode) return;
            if (!_alVwapSeries.length || !param.time || !param.point) {
                if (_alSelectedVwapIdx !== -1) _alDeselectAllVwaps();
                return;
            }
            var HIT_PX = 8;
            var hitIdx = -1;
            _alVwapSeries.forEach(function(entry, i) {
                if (!entry.dataMap) return;
                var avwapVal = entry.dataMap.get(param.time);
                if (avwapVal == null) return;
                var yCoord = entry.series.priceToCoordinate(avwapVal);
                if (yCoord == null) return;
                if (Math.abs(param.point.y - yCoord) <= HIT_PX) hitIdx = i;
            });
            if (hitIdx !== -1) {
                if (_alSelectedVwapIdx === hitIdx) { _alDeselectAllVwaps(); }
                else { _alSelectVwap(hitIdx); }
            } else {
                if (_alSelectedVwapIdx !== -1) _alDeselectAllVwaps();
            }
        });

        // OHLC legend
        var leg = document.createElement('div');
        leg.id = 'al-chart-legend';
        leg.style.cssText = 'position:absolute;top:8px;left:14px;z-index:10;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:#8b949e;pointer-events:none;line-height:1.8;background:rgba(13,17,23,0.85);padding:4px 10px;border-radius:4px;';
        container.style.position = 'relative';
        container.appendChild(leg);

        function fp(v) { return v != null ? v.toFixed(2) : '—'; }
        function fv(v) { return v==null?'—':v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':v.toFixed(0); }

        _alChart.subscribeCrosshairMove(function(p) {
            if (p.point && _alCandle) {
                var cursorPrice = _alCandle.coordinateToPrice(p.point.y);
                _alLastCrosshairPrice = (cursorPrice != null && !isNaN(cursorPrice)) ? cursorPrice : null;
            } else {
                _alLastCrosshairPrice = null;
            }
            _alLastCrosshairTime = p.time || null;
            if (!p.time || !p.seriesData || !p.seriesData.size) { leg.innerHTML = ''; return; }
            var d = p.seriesData.get(_alCandle); if (!d) { leg.innerHTML = ''; return; }
            var cl = d.close >= d.open ? '#089981' : '#b22833';
            var vd = p.seriesData.get(_alVol);
            var chgHtml = '';
            var barIdx = _barIdxByTime(_alOhlcv, p.time);
            if (barIdx > 0) {
                var prevClose = _alOhlcv[barIdx - 1].close;
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
        });

        // Market info bar
        (function() {
            if (!_alOhlcv.length) return;
            var last    = _alOhlcv[_alOhlcv.length - 1];
            var close   = last.close;
            var dayHigh = last.high;
            var dayLow  = last.low;
            var prevBar = _alOhlcv.length >= 2 ? _alOhlcv[_alOhlcv.length - 2] : null;
            var chg     = prevBar ? close - prevBar.close : 0;
            var pct     = prevBar && prevBar.close ? (chg / prevBar.close) * 100 : 0;
            var sliceLen = tf === 'W' ? 52 : tf === 'M' ? 12 : 252;
            var slice   = _alOhlcv.slice(Math.max(0, _alOhlcv.length - sliceLen));
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
            var adrEl = document.getElementById('al-chart-mkt-adr');
            if (adrEl) {
                var adrSd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var adrRaw = adrSd ? adrSd.adr_pct : null;
                if (adrRaw != null) {
                    adrEl.innerHTML = '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">ADR%</span>'
                                    + '<span style="color:#c9d1d9;font-size:12px;">' + adrRaw.toFixed(1) + '%</span>';
                    adrEl.style.display = 'inline-flex';
                } else { adrEl.style.display = 'none'; }
            }
            var mcapEl = document.getElementById('al-chart-mkt-mcap');
            if (mcapEl) {
                var sd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var mcapRaw = sd ? sd.MarketCap : null;
                if (mcapRaw != null) {
                    var mc = mcapRaw >= 1e12 ? (mcapRaw/1e12).toFixed(2)+'T' : mcapRaw >= 1e9 ? (mcapRaw/1e9).toFixed(2)+'B' : mcapRaw >= 1e6 ? (mcapRaw/1e6).toFixed(0)+'M' : mcapRaw;
                    mcapEl.innerHTML = '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">Mkt Cap</span><span style="color:#c9d1d9;font-size:12px;">' + mc + '</span>';
                    mcapEl.style.display = 'inline-flex';
                } else { mcapEl.style.display = 'none'; }
            }
            document.getElementById('al-chart-mkt-price').innerHTML =
                '<span style="color:#e6edf3;font-size:17px;font-weight:700;">' + fp(close) + '</span>' +
                '&nbsp;<span style="color:' + chgColor + ';font-size:13px;font-weight:600;">' + chgSign + fp(chg) + '&nbsp;(' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)</span>';
            document.getElementById('al-chart-mkt-day').innerHTML =
                '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">' + barLabel + '</span>' +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(dayLow) + '</span>' +
                mkBar(dayLow, dayHigh, close, 130, crLabel) +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(dayHigh) + '</span>';
            var w52HiPct   = (yrHigh > 0) ? (yrHigh - close) / yrHigh * 100 : 0;
            var w52HiLabel = yrHigh > 0 ? {
                text:  w52HiPct < 0.5 ? 'ATH' : ('-' + w52HiPct.toFixed(1) + '%'),
                color: w52HiPct <= 5 ? '#3fb950' : w52HiPct <= 15 ? '#e3852b' : '#f85149'
            } : null;
            document.getElementById('al-chart-mkt-52w').innerHTML =
                '<span style="color:#6e7681;font-size:11px;font-weight:600;letter-spacing:.04em;">52W</span>' +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(yrLow) + '</span>' +
                mkBar(yrLow, yrHigh, close, 120, w52HiLabel) +
                '<span style="color:#c9d1d9;font-size:12px;">' + fp(yrHigh) + '</span>';
            document.getElementById('al-chart-mkt-info').style.display = 'flex';
        })();

        // Keyboard: Delete/Escape for trendlines + AVWAP
        if (_alKeyHandler) { document.removeEventListener('keydown', _alKeyHandler); }
        _alKeyHandler = function(evt) {
            if (evt.key === 'Escape') {
                if (_alMeasureActive || _alMeasurePhase === 1) {
                    _alMeasureActive = false;
                    _alMeasurePhase  = 0;
                    if (_alMeasureRafId) { cancelAnimationFrame(_alMeasureRafId); _alMeasureRafId = null; }
                    document.removeEventListener('mousemove', _onAlMeasureDragMove);
                    document.removeEventListener('mouseup',   _onAlMeasureDragEnd);
                    document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
                }
                if (_alMeasureResult) {
                    _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
                    _alMeasureResult = null;
                }
                if (_alTrendDraw.active) {
                    _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
                    if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
                } else if (_alSelectedTrendlineIdx !== -1) {
                    _alDeselectAllTrendlines();
                } else if (_alSelectedVwapIdx !== -1) {
                    _alDeselectAllVwaps();
                }
                return;
            }
            if (evt.key !== 'Delete') return;
            if (_alSelectedTrendlineIdx !== -1) {
                evt.stopPropagation();
                var selIdx = _alSelectedTrendlineIdx;
                _alSelectedTrendlineIdx = -1;
                var selTl = _alTrendlines.splice(selIdx, 1)[0];
                try { if (_alCandle) _alCandle.detachPrimitive(selTl.primitive); } catch(e) {}
                return;
            }
            if (_alSelectedVwapIdx !== -1) {
                evt.stopPropagation();
                var selVwapIdx = _alSelectedVwapIdx;
                _alSelectedVwapIdx = -1;
                var removed = _alVwapSeries.splice(selVwapIdx, 1)[0];
                try { _alChart.removeSeries(removed.series); } catch(e) {}
                _alVwapSeries.forEach(function(entry) { entry.series.applyOptions({ lineWidth: 1.5 }); });
                return;
            }
            if (_alTrendlineMode && _alTrendlines.length) {
                evt.stopPropagation();
                var tLast = _alTrendlines.pop();
                try { if (_alCandle) _alCandle.detachPrimitive(tLast.primitive); } catch(e) {}
            }
        };
        document.addEventListener('keydown', _alKeyHandler);

        // Inject live bar
        _injectChartLiveBar(sym, tf, _alCandle, _alVol, _alOhlcv,
            function() { return _alSym !== sym || !_alCandle; });

        // Restore trendlines from alert store so they're visible when reviewing the chart
        alertsList.forEach(function(a) {
            if (a.alertType !== 'trendline' || a.ticker !== sym || !a.p1 || !a.p2) return;
            _addAlTrendline(a.p1, a.p2);
        });
    }

    // ── alSelectChart: open panel + fetch + build ─────────────────────────
    window.alSelectChart = function(ticker) {
        var panel = document.getElementById('al-chart-panel');
        if (!panel) return;
        panel.classList.add('open');

        // Header — symbol
        document.getElementById('al-chart-sym').textContent = ticker;

        // Meta — industry · rank
        var sd       = tickerMap && tickerMap[ticker];
        var industry = (sd && sd.industry) || '';
        var pct      = sd  && sd.Percentile != null ? sd.Percentile : null;
        var metaEl   = document.getElementById('al-chart-meta');
        if (metaEl) {
            var indRankHtml = '';
            if (industry && industriesData && industriesData.industries) {
                var indData = industriesData.industries.find(function(x){ return x.industry === industry; });
                var total   = industriesData.industries.length;
                if (indData && indData.rank != null) {
                    var rankPct   = indData.percentile != null ? indData.percentile : null;
                    var rankColor = rankPct != null ? (rankPct >= 75 ? '#3fb950' : rankPct >= 40 ? '#e3852b' : '#f85149') : '#6e7681';
                    indRankHtml = '<span class="meta-sep">·</span>' +
                        '<span style="color:' + rankColor + '">(' + indData.rank + '/' + total + ')</span>';
                }
            }
            metaEl.innerHTML = industry ? industryLinkHtml(industry, null) + indRankHtml : '';
        }

        // RS badges & fund stats
        applyRsBadge(document.getElementById('al-chart-rs-badge'), pct, sd ? sd.weighted_rs_pct : null, document.getElementById('al-chart-3mrs-badge'));
        var fsEl = document.getElementById('al-chart-fund-stats');
        if (fsEl) fsEl.innerHTML = fundStatsHtml(sd || null);

        // Details link
        var dBtn = document.getElementById('al-chart-details-btn');
        if (dBtn) { dBtn.href = 'https://finviz.com/quote.ashx?t=' + ticker.replace(/[^A-Z0-9]/gi, ''); dBtn.style.display = ''; }

        // Show settings bar + sync TF buttons
        var settingsBar = document.getElementById('al-chart-settings');
        if (settingsBar) settingsBar.style.display = 'flex';
        document.querySelectorAll('.al-chart-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === _alChartTf);
        });

        // Reset per-symbol tool state
        _alVwapMode = false; _alVwapSeries = []; _alSelectedVwapIdx = -1;
        var vwapBtn = document.getElementById('al-chart-vwap-btn');
        if (vwapBtn) vwapBtn.classList.remove('active');
        _alTrendlines = []; _alTrendlineFirst = null; _alSelectedTrendlineIdx = -1;
        if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
        _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
        var maPanel   = document.getElementById('al-chart-ma-panel');
        var maChevron = document.getElementById('al-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';

        _alVisibleBars = _alChartTf === 'D' ? 252 : _alChartTf === 'W' ? 104 : 60;
        _alSym = ticker;
        var widgetDiv = document.getElementById('al-chart-widget');
        widgetDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">Loading\u2026</div>';
        var loadTicker = ticker;
        fetchMcOhlcv(ticker, _alChartTf).then(function(ohlcv) {
            if (_alSym !== loadTicker) return;
            _buildAlChart(loadTicker, ohlcv, _alChartTf);
        });
    };

    window.alChartPanelClose = function() {
        var panel = document.getElementById('al-chart-panel');
        if (panel) panel.classList.remove('open');
        _destroyAlChart();
        var settingsBar = document.getElementById('al-chart-settings');
        if (settingsBar) settingsBar.style.display = 'none';
        var widgetDiv = document.getElementById('al-chart-widget');
        if (widgetDiv) widgetDiv.innerHTML = '';
    };

    // ── AL chart controls ─────────────────────────────────────────────────
    window.alChartSetTf = function(tf) {
        if (!_alSym) return;
        _alChartTf = tf;
        document.querySelectorAll('.al-chart-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        _alVwapMode = false; _alVwapSeries = []; _alSelectedVwapIdx = -1;
        var vwapBtn = document.getElementById('al-chart-vwap-btn');
        if (vwapBtn) vwapBtn.classList.remove('active');
        _alTrendlines = []; _alTrendlineFirst = null; _alSelectedTrendlineIdx = -1;
        if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
        _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
        _alMeasureMode = false; _alMeasureActive = false; _alMeasurePhase = 0; _alMeasureResult = null;
        if (_alMeasureRafId) { cancelAnimationFrame(_alMeasureRafId); _alMeasureRafId = null; }
        var alMBtn = document.getElementById('al-chart-measure-btn');
        if (alMBtn) alMBtn.classList.remove('active');
        document.removeEventListener('mousemove', _onAlMeasureDragMove);
        document.removeEventListener('mouseup',   _onAlMeasureDragEnd);
        document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
        var maPanel   = document.getElementById('al-chart-ma-panel');
        var maChevron = document.getElementById('al-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
        _alVisibleBars = tf === 'D' ? 252 : tf === 'W' ? 104 : 60;
        delete _mcOhlcvCache[_alSym + '_' + tf];
        var sym = _alSym;
        var container = document.getElementById('al-chart-widget');
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:12px;">Loading\u2026</div>';
        fetchMcOhlcv(sym, tf).then(function(ohlcv) {
            if (_alSym !== sym) return;
            _buildAlChart(sym, ohlcv, tf);
        });
    };

    window.alChartToggleMaPanel = function(e) {
        e.stopPropagation();
        var panel   = document.getElementById('al-chart-ma-panel');
        var chevron = document.getElementById('al-chart-ma-chevron');
        if (!panel) return;
        var opening = panel.style.display === 'none';
        panel.style.display = opening ? '' : 'none';
        if (chevron) chevron.style.transform = opening ? 'rotate(180deg)' : '';
        if (opening) {
            setTimeout(function() {
                function _outsideClick(ev) {
                    var wrap = document.getElementById('al-chart-ma-wrap');
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

    window.alChartToggleMa = function(key) {
        _alActiveMas[key] = !_alActiveMas[key];
        var btn = document.getElementById('al-chart-ma-' + key);
        if (btn) btn.classList.toggle('active', _alActiveMas[key]);
        if (!_alChart || !_alOhlcv.length) return;
        if (_alActiveMas[key]) {
            if (_alMaSeries[key]) return;
            var def = _MC_MA_DEFS[key]; if (!def) return;
            var s = _alChart.addSeries(LightweightCharts.LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false });
            var maData = _calcMA(_alOhlcv, key);
            s.setData(maData);
            _alMaSeries[key]  = s;
            _alMaDataMap[key] = new Map(maData.map(function(d) { return [d.time, d.value]; }));
        } else {
            if (_alMaSeries[key]) { try { _alChart.removeSeries(_alMaSeries[key]); } catch(e) {} delete _alMaSeries[key]; }
            delete _alMaDataMap[key];
        }
    };

    window.alChartToggleVwap = function() {
        _alVwapMode = !_alVwapMode;
        var btn = document.getElementById('al-chart-vwap-btn');
        if (btn) btn.classList.toggle('active', _alVwapMode);
        if (_alVwapMode && _alTrendlineMode) {
            _alTrendlineMode = false;
            var tBtn = document.getElementById('al-chart-trendline-btn');
            if (tBtn) tBtn.classList.remove('active');
            _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
            if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
        }
        if (_alVwapMode && _alMeasureMode) {
            _alMeasureMode = false;
            var mBtn = document.getElementById('al-chart-measure-btn');
            if (mBtn) mBtn.classList.remove('active');
        }
    };

    window.alChartToggleTrendline = function() {
        _alTrendlineMode = !_alTrendlineMode;
        var btn = document.getElementById('al-chart-trendline-btn');
        if (btn) btn.classList.toggle('active', _alTrendlineMode);
        if (_alTrendlineMode && _alVwapMode) {
            _alVwapMode = false;
            var vBtn = document.getElementById('al-chart-vwap-btn');
            if (vBtn) vBtn.classList.remove('active');
        }
        if (_alTrendlineMode && _alMeasureMode) {
            _alMeasureMode = false;
            var mBtn = document.getElementById('al-chart-measure-btn');
            if (mBtn) mBtn.classList.remove('active');
        }
        _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
        _alTrendlineFirst = null;
        if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
        if (_alSelectedTrendlineIdx !== -1) _alDeselectAllTrendlines();
    };

    window.alChartToggleMeasure = function() {
        _alMeasureMode = !_alMeasureMode;
        var btn = document.getElementById('al-chart-measure-btn');
        if (btn) btn.classList.toggle('active', _alMeasureMode);
        if (_alMeasureMode) {
            if (_alTrendlineMode) {
                _alTrendlineMode = false;
                var tBtn = document.getElementById('al-chart-trendline-btn');
                if (tBtn) tBtn.classList.remove('active');
                _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
                if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
            }
            if (_alVwapMode) {
                _alVwapMode = false;
                var vBtn = document.getElementById('al-chart-vwap-btn');
                if (vBtn) vBtn.classList.remove('active');
            }
        } else {
            if (_alMeasureActive || _alMeasurePhase === 1) {
                _alMeasureActive = false;
                _alMeasurePhase  = 0;
                if (_alMeasureRafId) { cancelAnimationFrame(_alMeasureRafId); _alMeasureRafId = null; }
                document.removeEventListener('mousemove', _onAlMeasureDragMove);
                document.removeEventListener('mouseup',   _onAlMeasureDragEnd);
                document.removeEventListener('mousemove', _onAlMeasurePreviewMove);
            }
            _hideMeasureOverlay(_alMeasureSvgOverlay, _alMeasureInfoDiv);
            _alMeasureResult = null;
        }
    };
    // ── END Alerts inline chart panel ─────────────────────────────────────

    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter') return;
        if (document.getElementById('al-modal-overlay').classList.contains('open')) alSubmitForm();
    });
    var _alConfirmCallback = null;

    window.alConfirmOpen = function(title, msg, callback, okLabel) {
        document.getElementById('al-confirm-title').textContent = title;
        document.getElementById('al-confirm-msg').textContent   = msg;
        document.getElementById('al-confirm-ok').textContent    = okLabel || 'Clear all';
        _alConfirmCallback = callback;
        document.getElementById('al-confirm-overlay').classList.add('open');
    };
    window.alConfirmClose = function() {
        document.getElementById('al-confirm-overlay').classList.remove('open');
        _alConfirmCallback = null;
    };
    window.alConfirmClear = function() {
        if (_alConfirmCallback) _alConfirmCallback();
        alConfirmClose();
    };
    window.alDeleteConfirm = function(idx, ticker) {
        alConfirmOpen(
            'Remove alert?',
            'The price alert for ' + ticker + ' will be permanently removed.',
            function() { alDelete(idx); },
            'Delete'
        );
    };
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') alConfirmClose();
    });
    // ── Alerts keyboard navigation ────────────────────────────────────────
    var _alKbIdx     = -1;
    var _alHistKbIdx = -1;
    var _alKbFocus   = 'alerts';

    function alKbGetAlertRows() {
        return Array.from(document.querySelectorAll('#al-list .al-row'))
            .filter(function(r) { return r.style.display !== 'none'; });
    }
    function alKbGetHistRows() {
        return Array.from(document.querySelectorAll('#al-hist-list .al-hist-row'));
    }
    function alKbSetAlertActive(rows, idx) {
        rows.forEach(function(r, i) { r.classList.toggle('al-row-active', i === idx); });
        if (rows[idx]) rows[idx].scrollIntoView({ block: 'nearest' });
    }
    function alKbSetHistActive(rows, idx) {
        rows.forEach(function(r, i) { r.classList.toggle('al-hist-row-active', i === idx); });
        if (rows[idx]) rows[idx].scrollIntoView({ block: 'nearest' });
    }
    function alKbTickerFromAlertRow(row) {
        var el = row && row.querySelector('.al-col-ticker');
        return el ? el.textContent.trim() : null;
    }
    function alKbTickerFromHistRow(row) {
        var el = row && row.querySelector('.al-hist-ticker');
        return el ? el.textContent.trim() : null;
    }

    // Track which list was last interacted with via click
    document.addEventListener('click', function(e) {
        if (currentView !== 'alerts') return;
        var isExcluded = e.target.closest('.al-col-del, .al-col-edit, .al-hist-del, .al-col-ticker-link, .al-hist-ticker');
        if (e.target.closest('#al-list')) {
            _alKbFocus = 'alerts';
            var rows = alKbGetAlertRows();
            var row  = e.target.closest('.al-row');
            if (row) {
                _alKbIdx = rows.indexOf(row);
                alKbSetAlertActive(rows, _alKbIdx);
                if (!isExcluded) {
                    var ticker = alKbTickerFromAlertRow(row);
                    if (ticker) alTickerClick(ticker);
                }
            }
        } else if (e.target.closest('#al-hist-list')) {
            _alKbFocus = 'history';
            var hRows = alKbGetHistRows();
            var hRow  = e.target.closest('.al-hist-row');
            if (hRow) {
                _alHistKbIdx = hRows.indexOf(hRow);
                alKbSetHistActive(hRows, _alHistKbIdx);
                if (!isExcluded) {
                    var hticker = alKbTickerFromHistRow(hRow);
                    if (hticker) alTickerClick(hticker);
                }
            }
        }
    }, true);

    document.addEventListener('keydown', function(e) {
        if (currentView !== 'alerts') return;
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown' && e.key !== 'Enter' && e.key !== 'Escape' && e.key !== 'Delete') return;
        var tag = document.activeElement && document.activeElement.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

        if (e.key === 'Escape') {
            alChartPanelClose();
            _alKbIdx     = -1;
            _alHistKbIdx = -1;
            alKbGetAlertRows().forEach(function(r) { r.classList.remove('al-row-active'); });
            alKbGetHistRows().forEach(function(r)  { r.classList.remove('al-hist-row-active'); });
            return;
        }

        if (e.key === 'Enter') {
            e.preventDefault();
            if (_alKbFocus === 'history') {
                var hRows  = alKbGetHistRows();
                var hticker = alKbTickerFromHistRow(hRows[_alHistKbIdx]);
                if (hticker) alOpenChart(hticker);
            } else {
                var rows   = alKbGetAlertRows();
                var aticker = alKbTickerFromAlertRow(rows[_alKbIdx]);
                if (aticker) alOpenChart(aticker);
            }
            return;
        }

        if (e.key === 'Delete') {
            if (_alKbFocus === 'history' && _alHistKbIdx >= 0) {
                e.preventDefault();
                var hRows = alKbGetHistRows();
                alHistDelete(_alHistKbIdx);
                var newRows = alKbGetHistRows();
                if (newRows.length) {
                    _alHistKbIdx = Math.min(_alHistKbIdx, newRows.length - 1);
                    alKbSetHistActive(newRows, _alHistKbIdx);
                } else {
                    _alHistKbIdx = -1;
                }
            } else if (_alKbFocus !== 'history' && _alKbIdx >= 0) {
                e.preventDefault();
                var rows = alKbGetAlertRows();
                var activeRow = rows[_alKbIdx];
                if (activeRow) {
                    var delBtn = activeRow.querySelector('.al-col-del');
                    if (delBtn) {
                        var match = delBtn.getAttribute('onclick').match(/alDeleteConfirm\((\d+),'([^']+)'\)/);
                        if (match) alDeleteConfirm(parseInt(match[1]), match[2]);
                    }
                }
            }
            return;
        }

        e.preventDefault();
        var dir = e.key === 'ArrowDown' ? 1 : -1;

        if (_alKbFocus === 'history') {
            var hRows = alKbGetHistRows();
            if (!hRows.length) return;
            _alHistKbIdx = Math.max(0, Math.min(hRows.length - 1, _alHistKbIdx + dir));
            alKbSetHistActive(hRows, _alHistKbIdx);
            var hticker = alKbTickerFromHistRow(hRows[_alHistKbIdx]);
            if (hticker) alSelectChart(hticker);
        } else {
            var rows = alKbGetAlertRows();
            if (!rows.length) return;
            if (_alKbIdx < 0) {
                // Sync from visually active row before snapping to first/last
                var _alActiveIdx = rows.findIndex(function(r) { return r.classList.contains('al-row-active'); });
                _alKbIdx = _alActiveIdx >= 0 ? _alActiveIdx : (dir === 1 ? 0 : rows.length - 1);
            } else _alKbIdx = Math.max(0, Math.min(rows.length - 1, _alKbIdx + dir));
            alKbSetAlertActive(rows, _alKbIdx);
            var aticker = alKbTickerFromAlertRow(rows[_alKbIdx]);
            if (aticker) alSelectChart(aticker);
        }
    });
    // ── END Alerts keyboard navigation ────────────────────────────────────

    // ── Alerts Multichart ─────────────────────────────────────────────────
    var alMcActive    = false;
    var alMcTimeframe = 'D';
    var alMcCols      = parseInt(localStorage.getItem('mcSharedCols') || '4');
    var alMcWidgets   = {};

    window.toggleAlMultichart = function() {
        alMcActive = !alMcActive;
        var btn    = document.getElementById('al-multichart-toggle-btn');
        var mcView = document.getElementById('al-multichart-view');
        var body   = document.querySelector('#view-alerts .al-body-wrap');
        btn.style.background  = alMcActive ? '#1f3a5c' : '';
        btn.style.borderColor = alMcActive ? '#388bfd' : '';
        btn.style.color       = alMcActive ? '#58a6ff' : '';
        if (mcView)  { mcView.style.display  = alMcActive ? 'flex' : 'none'; }
        if (body)    { body.style.display    = alMcActive ? 'none' : 'flex'; }
        if (alMcActive) renderAlMc();
    };

    window.setAlMcTf = function(tf) {
        alMcTimeframe = tf;
        document.querySelectorAll('#al-multichart-view .mc-tf-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-tf') === tf);
        });
        renderAlMc();
    };

    window.setAlMcCols = function(n) {
        alMcCols = n;
        document.querySelectorAll('#al-multichart-view .mc-col-btn').forEach(function(b){
            b.classList.toggle('active', +b.getAttribute('data-cols') === n);
        });
        document.getElementById('al-multichart-grid').style.gridTemplateColumns = 'repeat(' + n + ', 1fr)';
    };

    // ── Shared multichart column setter — syncs all 4 menus ──────────────────
    window.setSharedMcCols = function(n) {
        localStorage.setItem('mcSharedCols', String(n));
        if (window.setMcCols)      window.setMcCols(n);
        if (window.setScansMcCols) window.setScansMcCols(n);
        if (window.setWlMcCols)    window.setWlMcCols(n);
        if (window.setAlMcCols)    window.setAlMcCols(n);
    };

    // Apply stored col count to all button groups on load
    (function() {
        var stored = parseInt(localStorage.getItem('mcSharedCols') || '4');
        document.querySelectorAll('.mc-col-btn').forEach(function(b) {
            b.classList.toggle('active', +b.getAttribute('data-cols') === stored);
        });
    })();
    // ── END Shared multichart column setter ───────────────────────────────────

    function renderAlMc() {
        var grid = document.getElementById('al-multichart-grid');
        if (!grid) return;
        var seen = {}, tickers = [];
        alertsList.forEach(function(a) {
            if (!seen[a.ticker]) { seen[a.ticker] = true; tickers.push(a.ticker); }
        });
        _buildLwMcGrid(grid, tickers, alMcTimeframe, alMcCols, alMcWidgets, 'al');
    }

    // ── END PRICE ALERTS ─────────────────────────────────────────────────────
