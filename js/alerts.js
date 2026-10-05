    // ── PRICE ALERTS ─────────────────────────────────────────────────────────
    var alertsList       = [];   // [{ticker, condition:'above'|'below', price, addedAt}]
    var alSortKey = 'away';  // 'away' | 'added' | 'chgpct' | null
    var alSortDir = 'asc';   // 'asc' | 'desc'
    var alertFiredList   = [];   // [{ticker, condition, alertPrice, hitPrice, firedAt, dismissed}]
    var alertPrices      = {};   // {ticker: latestPrice}
    var alertPrevClose   = {};   // {ticker: prevClosePrice}
    var alertDayHigh     = {};   // {ticker: today's intraday high} — used to catch spikes the 10s poll misses
    var alertDayLow      = {};   // {ticker: today's intraday low}  — same, for 'below' conditions
    var alertPriceAt     = {};   // {ticker: Date.now() of the last quote we actually received} — line alerts refuse to act on a stale price
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

    var _alLoaded      = false; // guard: once the user has mutated alerts, ignore any late alLoad responses
    var _alFiredLoaded = false; // guard: once fired-history has been mutated, ignore any late alLoad responses

    // ── Canonical alert identity ─────────────────────────────────────────────────────────────
    // ONE function builds the dedup key for every alert type, for live alerts AND fired-history
    // entries (they share field names). It used to be copy-pasted in 7+ places that drifted apart
    // (alDelete and the edit form never learned about trendline/AVWAP, so deleting one left the
    // real key behind). Never build a key inline again -- call _alKey().
    //
    // Trendline keys include the anchor PRICES, not just the anchor dates: two different lines can
    // share both anchor dates (e.g. after a vertical-only drag), and they must not collapse into one.
    function _alToUnix(t) {
        if (t == null) return null;
        if (typeof t === 'number') return isFinite(t) ? t : null;
        if (typeof t === 'string') { var ms = new Date(t).getTime(); return isNaN(ms) ? null : Math.floor(ms / 1000); }
        if (t.year != null) return Math.floor(Date.UTC(t.year, t.month - 1, t.day) / 1000);
        return null;
    }
    function _alPtUnix(p) { return p ? ((p.unix != null) ? p.unix : _alToUnix(p.time)) : null; }
    function _alAnchorUnix(a) { return (a.anchorUnix != null) ? a.anchorUnix : _alToUnix(a.anchorTime); }
    function _alPtKey(p) {
        var u = _alPtUnix(p);
        return (u == null ? '' : u) + '@' + ((p && p.price != null && isFinite(p.price)) ? Number(p.price).toFixed(4) : '');
    }
    function _alKey(a) {
        if (a.alertType === 'macross') return a.ticker + '_macross_' + a.ma1Key + '_' + a.ma2Key + '_' + a.condition;
        if (a.alertType === 'ma') return a.ticker + '_ma_' + a.maKey + '_' + a.condition;
        if (a.alertType === 'pattern') return window.alPatternAlertKey(a);
        if (a.alertType === 'trendline') return a.ticker + '_trendline_' + _alPtKey(a.p1) + '_' + _alPtKey(a.p2) + '_' + a.condition;
        if (a.alertType === 'avwap') { var au = _alAnchorUnix(a); return a.ticker + '_avwap_' + (au == null ? '' : au) + '_' + a.condition; }
        // Plain/RSI alerts: live alerts carry .price, fired-history entries carry .alertPrice
        return a.ticker + '_' + (a.price != null ? a.price : a.alertPrice) + '_' + a.condition;
    }
    function _alFiredHistKey(f) { return _alKey(f); } // kept under its old name for existing callers

    // Rebuild the "already fired this session" map from persisted history on page load.
    // The old version marked EVERY history entry's key as fired. If you fired an alert and then
    // re-armed the same ticker/anchor/condition, the history entry re-poisoned the key after the next
    // reload and the re-armed alert was skipped forever (shown as "Fired" though it never had).
    // A live alert with the same key that was added AFTER that entry fired is a deliberate re-arm,
    // so it must not be suppressed. (The suppression still covers the case it exists for: an alert
    // that fired but whose removal failed to persist is older than its own history entry.)
    function _alRehydrateFiredSess() {
        alertFiredList.forEach(function(f) {
            var k = _alFiredHistKey(f);
            var firedMs = f.firedAt ? new Date(f.firedAt).getTime() : NaN;
            var rearmed = !isNaN(firedMs) && alertsList.some(function(a) {
                if (!a.addedAt || _alKey(a) !== k) return false;
                var addedMs = new Date(a.addedAt).getTime();
                return !isNaN(addedMs) && addedMs > firedMs;
            });
            if (!rearmed) _alertFiredSess[k] = true;
        });
    }

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
            // Guard each list independently — alSave() marks _alLoaded, alSaveFired() marks _alFiredLoaded,
            // so a user mutation to active alerts no longer blocks fired history from being restored from KV.
            var didChange = false;
            if (!_alLoaded) {
                _alLoaded = true;
                var rawAlerts = r[0] || localStorage.getItem(LS_AL_KEY);
                try { alertsList = rawAlerts ? JSON.parse(rawAlerts) : []; } catch(e) { alertsList = []; }
                // Mirror KV data to localStorage so fallback stays fresh
                if (r[0]) { try { localStorage.setItem(LS_AL_KEY, r[0]); } catch(e) {} }
                didChange = true;
            }
            if (!_alFiredLoaded) {
                _alFiredLoaded = true;
                var rawFired = r[1] || localStorage.getItem(LS_AL_FIRED_KEY);
                try { alertFiredList = rawFired ? JSON.parse(rawFired) : []; } catch(e) { alertFiredList = []; }
                // Mirror KV data to localStorage so fallback stays fresh
                if (r[1]) { try { localStorage.setItem(LS_AL_FIRED_KEY, r[1]); } catch(e) {} }
                _alRehydrateFiredSess();
                didChange = true;
            }
            if (!didChange) return;
            alUpdateBadge();
            alStartBackgroundPolling();
            renderHistory();
            if (currentView === 'alerts') renderAlerts();
            alStampBadges();
        }).catch(function() {
            // KV totally unavailable — load from localStorage, same independent guards
            var didChange = false;
            if (!_alLoaded) {
                _alLoaded = true;
                try { alertsList = JSON.parse(localStorage.getItem(LS_AL_KEY) || '[]'); } catch(e) { alertsList = []; }
                didChange = true;
            }
            if (!_alFiredLoaded) {
                _alFiredLoaded = true;
                try { alertFiredList = JSON.parse(localStorage.getItem(LS_AL_FIRED_KEY) || '[]'); } catch(e) { alertFiredList = []; }
                _alRehydrateFiredSess();
                didChange = true;
            }
            if (!didChange) return;
            alUpdateBadge();
            alStartBackgroundPolling();
            renderHistory();
            if (currentView === 'alerts') renderAlerts();
            alStampBadges();
        });
    }

    var _alLastSaveAt = 0, _alSaveSoonTimer = null;
    // Saves that come from the ENGINE (arming state, baselines, removal of a fired alert) rather than from a click.
    // Workers KV allows about one write per second per key (see the note in alCheckTriggers), and a creation click
    // has just written the same 'price_alerts' key: if we're inside that window, queue ONE trailing write (it saves
    // whatever the latest state is) instead of racing it.
    function _alSaveSoon() {
        var wait = 1100 - (Date.now() - _alLastSaveAt);
        if (wait <= 0) { alSave(); return; }
        if (_alSaveSoonTimer) return;
        _alSaveSoonTimer = setTimeout(function() { _alSaveSoonTimer = null; alSave(); }, wait + 50);
    }
    function alSave() {
        _alLoaded = true; // mark as user-owned so any late alLoad response won't overwrite
        _alLastSaveAt = Date.now();
        var str = JSON.stringify(alertsList);
        kvSet('price_alerts', str);
        try { localStorage.setItem(LS_AL_KEY, str); } catch(e) {}
        alStampBadges();
    }

    var _AL_BELL_SVG = '<svg width="8" height="9" viewBox="0 0 8 9" fill="none" style="flex-shrink:0;display:block;"><path d="M4 1a2 2 0 0 1 2 2v1.5l.8 1H1.2L2 4.5V3A2 2 0 0 1 4 1zm-1 5.5h2" stroke="var(--warning-alt)" stroke-width="1.1" stroke-linecap="round"/></svg>';
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
        _alFiredLoaded = true; // mark fired-history as user-owned so any late alLoad response won't overwrite
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
        }, 10 * 1000);
    }

    // ── Daily history + line evaluation for trendline and AVWAP alerts ───────────────────────────────
    //
    // Both alert types are computed from the per-ticker daily OHLCV series that multichart.js keeps in
    // _mcOhlcvCache['<TICKER>_D']. Everything that decides "what is this alert's line worth right now"
    // lives in this section and nowhere else, so the engine (alCheckTriggers), the table (Away column and
    // sort), the chart restore and the history view can't drift apart.
    //
    // Guarantees (each of these used to be a bug):
    //  - The cached series is only ever REPLACED by a successful refetch, never deleted first, so a failed
    //    refetch can't throw away ten years of good history.
    //  - Refreshing is fire-and-forget: alFetchPrices never waits on it, so a slow or rate-limited history
    //    fetch can't delay the evaluation of any alert (price alerts included).
    //  - Refetches are throttled with backoff. A ticker that never gets a bar for "today" (halted,
    //    illiquid, a holiday on that exchange) is retried at 15s, 30s, 60s ... capped at 15 min, not every poll.
    //  - Today's bar is kept current in place: close/high/low from the quote (including the quote's own
    //    dayHigh/dayLow) every poll, and volume from a throttled refetch (quotes_batch carries no volume).
    //    A frozen volume under-weights today in the AVWAP all session -- worst for short anchors.
    //  - Trendlines are refreshed once per ET trading day, and "today" is placed on the trading-day axis from
    //    the calendar (_alNowSlot), so a cache that is a few days stale can't shift the line either.

    var _alFetchFailedAt = {};
    var AL_REFETCH_COOLDOWN_MS = 5 * 60 * 1000;     // after a FAILED refetch, leave that ticker alone this long
    var AL_REFRESH_ACTIVE_MS   = 5 * 60 * 1000;     // while today's bar is present, re-pull it this often (volume)
    var AL_REFRESH_RETRY_MIN   = 15 * 1000;         // today's bar still missing: first retry after this ...
    var AL_REFRESH_RETRY_MAX   = 15 * 60 * 1000;    // ... doubling up to this cap
    var AL_QUOTE_MAX_AGE_MS    = 90 * 1000;         // a line alert won't act on a quote older than this
    var _alRefresh    = {};   // ticker -> { day, nextAt, gap, inflight }
    var _alEvalStatus = {};   // alert key -> { warn, msg }  (runtime only): why an alert isn't being evaluated right now

    function _alShouldSkipRefetch(ticker) {
        var failedAt = _alFetchFailedAt[ticker];
        return !!failedAt && (Date.now() - failedAt) < AL_REFETCH_COOLDOWN_MS;
    }

    // ── calendar / trading-day axis ──
    var _alEtFmt = null;
    // The current New York calendar date as an epoch-day number (days since 1970-01-01).
    function _alEtDayNum() {
        try {
            if (!_alEtFmt) _alEtFmt = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });
            var y = 0, m = 0, d = 0;
            _alEtFmt.formatToParts(new Date(Date.now())).forEach(function(p) {
                if (p.type === 'year') y = +p.value; else if (p.type === 'month') m = +p.value; else if (p.type === 'day') d = +p.value;
            });
            return Math.floor(Date.UTC(y, m - 1, d) / 86400000);
        } catch (e) { return Math.floor(Date.now() / 86400000); }
    }
    // Number of weekdays (Mon-Fri) in the half-open range (d0, d1]; both are epoch-day numbers. Holidays
    // aren't known here, so a holiday counts as a slot (off by at most one slot per holiday inside the span).
    function _alWeekdaysBetween(d0, d1) {
        var c = 0;
        for (var d = d0 + 1; d <= d1; d++) { var w = (d + 4) % 7; if (w !== 0 && w !== 6) c++; }
        return c;
    }
    function _alDailyCache(ticker) {
        var arr = _mcOhlcvCache[ticker + '_D'] || _mcOhlcvCache[ticker + '_d'];
        return (arr && arr.length) ? arr : null;
    }
    function _alHasTodayBar(ticker) {
        var arr = _alDailyCache(ticker);
        return !!arr && Math.floor(arr[arr.length - 1].time / 86400) >= _alEtDayNum();
    }
    // Position of an epoch-day on the trading-day axis the chart draws on (an index into the daily series).
    //   past/present date -> index of the first bar on or after it (exact for daily bars; weekly/monthly
    //                        stamps can land on a non-trading day), or -1 if older than the loaded history
    //   future date       -> last index + number of weekdays past the last bar (a weekend date rolls to Monday)
    function _alSlotOfDay(ohlcv, day) {
        var n = ohlcv.length;
        var lastDay = Math.floor(ohlcv[n - 1].time / 86400);
        if (day > lastDay) {
            var slots = _alWeekdaysBetween(lastDay, day), w = (day + 4) % 7;
            if (w === 0 || w === 6) slots += 1;
            return (n - 1) + slots;
        }
        if (Math.floor(ohlcv[0].time / 86400) > day) return -1;
        var lo = 0, hi = n - 1;
        while (lo < hi) { var mid = (lo + hi) >> 1; if (Math.floor(ohlcv[mid].time / 86400) >= day) hi = mid; else lo = mid + 1; }
        return lo;
    }
    // Where "now" sits on that axis: today's bar if it exists, otherwise the calendar-correct slot past the
    // last bar -- so a tab that has been open for days with a stale cache still evaluates at the right slot.
    function _alNowSlot(ohlcv) {
        var n = ohlcv.length, lastDay = Math.floor(ohlcv[n - 1].time / 86400), today = _alEtDayNum();
        return (n - 1) + (lastDay >= today ? 0 : _alWeekdaysBetween(lastDay, today));
    }

    // ── line values ──
    // A trendline is linear in trading-day slots (that is how the chart draws it: Lightweight Charts gives
    // weekends/holidays zero width). Evaluated at today's slot. Returns { v, why }; v is null when it can't be
    // computed, and `why` says so in words (the table shows it instead of silently looking "Active").
    function _alTrendlineEval(ticker, p1, p2) {
        var ohlcv = _alDailyCache(ticker);
        if (!ohlcv) return { v: null, why: 'Daily history not loaded yet' };
        var u1 = _alPtUnix(p1), u2 = _alPtUnix(p2);
        if (u1 == null || u2 == null || p1.price == null || p2.price == null) return { v: null, why: 'Line points are invalid' };
        var i1 = _alSlotOfDay(ohlcv, Math.floor(u1 / 86400)), i2 = _alSlotOfDay(ohlcv, Math.floor(u2 / 86400));
        if (i1 < 0 || i2 < 0) return { v: null, why: 'A line anchor is older than the loaded history' };
        if (i1 === i2) return { v: null, why: 'Both line anchors fall on the same trading day' };
        var v = p1.price + (p2.price - p1.price) * (_alNowSlot(ohlcv) - i1) / (i2 - i1);
        return isFinite(v) ? { v: v, why: null } : { v: null, why: 'Line value is not a number' };
    }
    function _alTrendlineValueNow(ticker, p1, p2) { return _alTrendlineEval(ticker, p1, p2).v; }

    // AVWAP from the anchor day (first daily bar on/after it) to the latest bar. Returns { line, why }.
    function _alAvwapEval(a) {
        var ohlcv = _alDailyCache(a.ticker);
        if (!ohlcv) return { line: null, why: 'Daily history not loaded yet' };
        var t = _alAnchorUnix(a);
        if (t == null) return { line: null, why: 'Anchor date is missing' };
        var idx = _alSlotOfDay(ohlcv, Math.floor(t / 86400));
        if (idx < 0) return { line: null, why: 'Anchor is older than the loaded history' };
        if (idx > ohlcv.length - 1) return { line: null, why: 'Anchor is in the future' };
        var data = _calcAVWAP(ohlcv, idx);
        if (!data.length) return { line: null, why: 'No volume data since the anchor' };
        return { line: data[data.length - 1].value, why: null };
    }
    // What is this trendline/AVWAP alert's line worth right now?  -> { line: number|null, why: string|null }
    function _alLineLevel(a) {
        if (a.alertType === 'trendline') {
            if (!a.p1 || !a.p2) return { line: null, why: 'Line points are missing' };
            var r = _alTrendlineEval(a.ticker, a.p1, a.p2);
            return { line: r.v, why: r.why };
        }
        return _alAvwapEval(a);
    }

    // Distance in % between the live price and a trendline/AVWAP line, or null when it can't be computed.
    // Used by BOTH the Away column and the Away sort, so they can't disagree (the sort used to have no AVWAP branch).
    function _alLineAwayPct(a) {
        var px = alertPrices[a.ticker];
        if (px == null) return null;
        var lv = _alLineLevel(a);
        return (lv.line != null && lv.line > 0) ? Math.abs((px - lv.line) / lv.line * 100) : null;
    }
    // Status pill for the table. An alert that can't currently be evaluated says so (and why) instead of looking "Active".
    function _alStatusPillHtml(fired, key) {
        if (fired) return '<span class="al-pill al-pill-fired">Fired</span>';
        var st = _alEvalStatus[key];
        if (st && st.warn) return '<span class="al-pill al-pill-active al-pill-warn" style="color:var(--warning-alt);" title="' + esc(st.msg) + '">Not evaluating</span>';
        if (st) return '<span class="al-pill al-pill-active" title="' + esc(st.msg) + '">Active</span>';
        return '<span class="al-pill al-pill-active">Active</span>';
    }

    // On creation: if the price is already known to be on the far side of the line, arm right away, so a gap
    // through the line at the next open still counts. Otherwise the first evaluation arms it.
    function _alTryArm(a) {
        var lv = _alLineLevel(a);
        if (lv.line == null) return;
        var px = alertPrices[a.ticker];
        if (px == null) { var arr = _alDailyCache(a.ticker); if (arr) px = arr[arr.length - 1].close; }
        if (px == null) return;
        if (a.condition === 'above' ? px < lv.line : px > lv.line) a.armed = true;
    }

    function _alNoEval(key, warn, msg) {
        if (msg) _alEvalStatus[key] = { warn: !!warn, msg: msg }; else delete _alEvalStatus[key];
        return null;
    }
    // Decide whether a trendline / AVWAP alert fires on this tick.
    //
    // These fire on a CROSS, not on a state. An alert is "armed" only once it has seen the price on the far side
    // of the line, and only an armed alert can fire when the price reaches or passes the line. Before this, an
    // "above" alert created while the price was already above fired on the very next poll, and any error in the line
    // value (stale cache, stale quote) turned straight into a false alert. It's the same idea as the baseline the
    // price-alert branch already uses.
    //
    // Firing also requires: the session is open, a quote received within AL_QUOTE_MAX_AGE_MS, and today's daily bar
    // present in the cache. That last one is what keeps a market holiday from firing alerts off the stale last
    // close, whether or not wlIsMarketOpen() knows about holidays. Outside those conditions the alert only observes
    // (it can still arm from the last close, so a gap through the line at the next open is caught).
    //
    // Returns null when the alert can't be evaluated right now (the reason is left in _alEvalStatus for the table),
    // else { hit, line, price, changed } where `changed` means a.armed was modified and must be saved.
    function _alLineEval(a, key, mktOpen) {
        var price = alertPrices[a.ticker];
        if (price == null) return _alNoEval(key, true, 'Waiting for a quote');
        var lv = _alLineLevel(a);
        if (lv.line == null) return _alNoEval(key, true, lv.why);
        var quoteAge = Date.now() - (alertPriceAt[a.ticker] || 0);
        if (mktOpen && quoteAge > AL_QUOTE_MAX_AGE_MS)
            return _alNoEval(key, true, 'Quote is stale (' + Math.round(quoteAge / 1000) + 's old), not evaluating');
        var changed = false;
        // Alerts saved before arming existed keep their old level-triggered behaviour: mark them armed.
        if (a.armed === undefined) { a.armed = true; changed = true; }
        var above      = a.condition === 'above';
        var onHitSide  = above ? price >= lv.line : price <= lv.line;
        var onFarSide  = above ? price <  lv.line : price >  lv.line;
        if (!a.armed) {
            if (onFarSide) { a.armed = true; changed = true; }
            else _alEvalStatus[key] = { warn: false, msg: 'Not armed yet: waits for price to move to the other side of the line, then fires when it crosses' };
            if (a.armed) delete _alEvalStatus[key];
            return { hit: false, line: lv.line, price: price, changed: changed };
        }
        if (!mktOpen) { delete _alEvalStatus[key]; return { hit: false, line: lv.line, price: price, changed: changed }; }
        if (!_alHasTodayBar(a.ticker)) {
            _alEvalStatus[key] = { warn: false, msg: "Waiting for today's trading data" };
            return { hit: false, line: lv.line, price: price, changed: changed };
        }
        delete _alEvalStatus[key];
        return { hit: onHitSide, line: lv.line, price: price, changed: changed };
    }

    // ── keeping the daily series fresh ──
    // Today's cached bar, kept current in place. Close and high/low come from the quote we already fetched
    // (including the quote's own day range, so a spike between two polls isn't lost). Only while the session is
    // open: after hours, Yahoo's own daily bar with the official close beats an extended-hours print.
    function _alPatchTodayBar(ticker, live) {
        var arr = _alDailyCache(ticker);
        if (!arr) return;
        var bar = arr[arr.length - 1];
        if (Math.floor(bar.time / 86400) < _alEtDayNum()) return;   // not today's bar
        if (live != null && isFinite(live)) {
            bar.close = live;
            if (live > bar.high) bar.high = live;
            if (live < bar.low)  bar.low  = live;
        }
        var hi = alertDayHigh[ticker], lo = alertDayLow[ticker];
        if (hi != null && isFinite(hi) && hi > bar.high) bar.high = hi;
        if (lo != null && isFinite(lo) && lo > 0 && lo < bar.low) bar.low = lo;
    }

    function _alRefreshDaily(ticker, st, force) {
        st.inflight = true;
        return fetchMcOhlcv(ticker, 'D', false, force).then(function(ohlcv) {
            st.inflight = false;
            var now = Date.now();
            if (!ohlcv || !ohlcv.length) {              // failed or empty: keep whatever is cached, cool this ticker down
                _alFetchFailedAt[ticker] = now;
                return false;
            }
            delete _alFetchFailedAt[ticker];
            st.day = _alEtDayNum();
            if (_alHasTodayBar(ticker)) { st.gap = AL_REFRESH_RETRY_MIN; st.nextAt = now + AL_REFRESH_ACTIVE_MS; }
            else { st.nextAt = now + st.gap; st.gap = Math.min(st.gap * 2, AL_REFRESH_RETRY_MAX); }
            return true;
        }, function() { st.inflight = false; _alFetchFailedAt[ticker] = Date.now(); return false; });
    }

    // Called from every price poll. Never awaited by the caller; never throws into it.
    function alSyncLineData() {
        var need = {};   // ticker -> true if an AVWAP alert needs it (volume matters), false if only trendlines do
        alertsList.forEach(function(a) {
            if (a.alertType === 'avwap') need[a.ticker] = true;
            else if (a.alertType === 'trendline' && !need[a.ticker]) need[a.ticker] = false;
        });
        var mktOpen = wlIsMarketOpen(), now = Date.now(), todayKey = _alEtDayNum();
        Object.keys(need).forEach(function(ticker) {
            var st = _alRefresh[ticker] || (_alRefresh[ticker] = { day: null, nextAt: 0, gap: AL_REFRESH_RETRY_MIN, inflight: false });
            if (st.inflight) return;
            var arr = _alDailyCache(ticker);
            if (arr && mktOpen) _alPatchTodayBar(ticker, alertPrices[ticker]);
            if (_alShouldSkipRefetch(ticker)) return;
            var want = false, force = !!arr;
            if (!arr) want = true;                                    // nothing cached yet
            else if (st.day !== todayKey) want = true;                // first look today: pick up yesterday's final bar / today's bar
            else if (mktOpen && now >= st.nextAt) {
                var lastDay = Math.floor(arr[arr.length - 1].time / 86400);
                if (lastDay < todayKey) want = true;                  // today's bar still missing (backoff'd retry)
                else if (need[ticker]) want = true;                   // AVWAP: re-pull today's volume
            }
            if (!want) return;
            _alRefreshDaily(ticker, st, force).then(function(ok) {
                if (!ok) return;
                try { alCheckTriggers(); renderAlerts(); } catch (e) {}   // table + arming reflect the fresh data right away
            });
        });
    }

    function alFetchPrices() {
        var activeTickers  = alertsList.map(function(a) { return a.ticker; });
        var historyTickers = alertFiredList.map(function(f) { return f.ticker; });
        var tickers = activeTickers.concat(historyTickers)
            .filter(function(v, i, arr) { return arr.indexOf(v) === i; });
        if (!tickers.length) return Promise.resolve();
        // Batch size MUST match the cap in the yahoo-proxy Worker's quotes_batch
        // handler (currently tickersParam...slice(0, 50)). The Worker silently
        // drops anything past its cap with no error and a normal 200 response,
        // so a mismatch here doesn't fail loudly — it just quietly returns fewer
        // quotes than requested. If that Worker-side cap ever changes, this
        // number needs to change with it. Raised from 30 — that was the
        // Yahoo-era CPU-budget limit — but deliberately stopped at 50 rather
        // than the confirmed 20 req/sec ceiling's full headroom, since
        // Questrade doesn't document a max ids-per-call limit and this
        // hasn't been verified against the live API yet.
        var AL_QUOTE_BATCH_SIZE = 50;
        var batches = [];
        for (var i = 0; i < tickers.length; i += AL_QUOTE_BATCH_SIZE) batches.push(tickers.slice(i, i + AL_QUOTE_BATCH_SIZE));
        return Promise.all(batches.map(function(batch) {
            var url = WL_PROXY + '?action=quotes_batch&tickers=' + batch.map(encodeURIComponent).join(',');
            return fetch(url).then(function(r) { return r.ok ? r.json() : null; }).catch(function() { return null; });
        })).then(function(results) {
            results.forEach(function(data) {
                if (!data || !data.quotes) return;
                data.quotes.forEach(function(q) {
                    if (!q) return;
                    // quotes_batch falls back to Yahoo when Questrade has nothing
                    // live (market closed). Yahoo's shape is {symbol, price: null,
                    // regularMarketPrice} instead of {ticker, price} — without this
                    // fallback the quote gets silently dropped and the price never
                    // updates for any ticker that only ever got fetched while closed.
                    var qTicker = q.ticker || q.symbol;
                    var qPrice  = q.price != null ? q.price : q.regularMarketPrice;
                    if (qTicker && qPrice != null) {
                        alertPrices[qTicker]    = qPrice;
                        alertPriceAt[qTicker]   = Date.now();
                        // prevClose now comes from the daily snapshot's preserved
                        // close (tickerMap[ticker]._snapPrice), not the Worker
                        // response — Questrade quotes don't include one.
                        var snapRow = tickerMap && tickerMap[qTicker];
                        alertPrevClose[qTicker] = (snapRow && snapRow._snapPrice) || null;
                        alertDayHigh[qTicker]   = (q.dayHigh != null) ? q.dayHigh : null;
                        alertDayLow[qTicker]    = (q.dayLow  != null) ? q.dayLow  : null;
                    }
                });
            });
            alUpdateEstimatedMAs();
            // Daily history for trendline/AVWAP alerts refreshes in the background. It is deliberately NOT
            // awaited: a slow, failing or rate-limited history fetch must never delay evaluating any alert.
            try { alSyncLineData(); } catch (e) {}
        }).then(function() {
            alCheckTriggers();
            // Not gated on currentView, unlike the other renderAlerts() call
            // sites in this file — this is the repeating price-poll path,
            // and after-hours it may be the ONLY fetch that runs all night
            // (polling stops once the market closes). Gating it here meant
            // correctly-fetched prices could sit in memory with nothing
            // ever telling the table to redraw. renderAlerts() already
            // no-ops immediately if the alerts DOM isn't present, so this
            // is safe to call unconditionally.
            renderAlerts();
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
        var anyBaselineChanged = false;
        var mktOpen = wlIsMarketOpen();
        // The snapshot is indexed once per check (it used to be fully scanned once per alert, per poll,
        // including for alert types that never read it). First match wins, as before.
        var snapIdx = null;
        function snapLookup(ticker) {
            if (!snapshot || !snapshot.by_industry) return null;
            if (!snapIdx) {
                snapIdx = Object.create(null);
                for (var ind in snapshot.by_industry) {
                    var stocks = snapshot.by_industry[ind];
                    for (var si = 0; si < stocks.length; si++) { var tk = stocks[si].ticker; if (!(tk in snapIdx)) snapIdx[tk] = stocks[si]; }
                }
            }
            return snapIdx[ticker] || null;
        }
        alertsList.forEach(function(a) {
            var key = _alKey(a);
            if (_alertFiredSess[key]) return;
            var hit = false;
            var hitVal = null;
            var lineAtFire = null;   // trendline/AVWAP only: the line's value at the moment it fired

            // This ticker's snapshot row (only the snapshot-driven alert types read it)
            var stockData = (a.alertType === 'macross' || a.alertType === 'ma' || a.alertType === 'rsi14' || a.alertType === 'pattern')
                ? snapLookup(a.ticker) : null;

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
            } else if (a.alertType === 'trendline' || a.alertType === 'avwap') {
                // Both line types share one evaluator: it fires on a CROSS (armed -> reaches the line), refuses to
                // act on a stale quote / closed session / missing today's bar, and records WHY when it can't evaluate.
                var lr = _alLineEval(a, key, mktOpen);
                if (!lr) return;
                if (lr.changed) anyBaselineChanged = true;   // arming state changed: persist (still one write per pass)
                if (!lr.hit) return;
                hit = true;
                hitVal = lr.price;
                lineAtFire = lr.line;
            } else {
                var price = alertPrices[a.ticker];
                var dHigh = alertDayHigh[a.ticker];
                var dLow  = alertDayLow[a.ticker];
                if (price == null) return;
                // Baseline = the day's high/low as of the first time we ever checked
                // this alert, reset at the start of each new trading day. Only a move
                // *past* the baseline counts as something that happened while the
                // alert was actually watching -- see the spike-catch comment below for
                // why we track a high/low at all instead of just the live price.
                var todayDay = Math.floor(Date.now() / 86400000);
                var baselineChanged = false;
                if (a.baselineDay !== todayDay) {
                    a.baselineDay = todayDay;
                    a.baselineDayHigh = dHigh;
                    a.baselineDayLow  = dLow;
                    baselineChanged = true;
                } else {
                    if (a.baselineDayHigh == null && dHigh != null) { a.baselineDayHigh = dHigh; baselineChanged = true; }
                    if (a.baselineDayLow  == null && dLow  != null) { a.baselineDayLow  = dLow;  baselineChanged = true; }
                }
                if (baselineChanged) anyBaselineChanged = true;
                // Compare against the day's high/low, not just the last-polled price.
                // A 10s poll can otherwise completely miss a spike that prints and
                // reverts faster than the poll interval (e.g. high $313.33, alert at
                // $309.61, price already back to $307.80 by the next poll — the
                // condition was met on the exchange but never observed by us). But
                // only count a high/low that's past the baseline above -- otherwise a
                // brand new alert set after a big move already happened that day would
                // fire immediately off history that predates the alert.
                if (a.condition === 'above') {
                    var newHigh = (dHigh != null && a.baselineDayHigh != null && dHigh > a.baselineDayHigh) ? dHigh : null;
                    hitVal = (newHigh != null) ? Math.max(price, newHigh) : price;
                    hit = hitVal >= a.price;
                } else {
                    var newLow = (dLow != null && a.baselineDayLow != null && dLow < a.baselineDayLow) ? dLow : null;
                    hitVal = (newLow != null) ? Math.min(price, newLow) : price;
                    hit = hitVal <= a.price;
                }
            }
            if (!hit) return;
            _alertFiredSess[key] = true;
            alertFiredList.unshift({
                ticker: a.ticker, condition: a.condition,
                alertPrice: (a.alertType === 'trendline' || a.alertType === 'avwap') ? null : a.price, hitPrice: hitVal,
                lineValue: lineAtFire,
                alertType: a.alertType || 'price',
                maKey: a.maKey || null,
                ma1Key: a.ma1Key || null, ma2Key: a.ma2Key || null,
                patternKeys: window.alGetPatternKeys(a), patternTf: a.patternTf || null,
                triggeredPatternKeys: (a.alertType === 'pattern' ? triggeredPats : null),
                p1: a.p1 || null, p2: a.p2 || null,
                anchorTime: a.anchorTime || null, anchorUnix: a.anchorUnix || null,
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
                    body = (a.condition === 'above' ? '▲ above' : '▼ below') + ' trendline $' + lineAtFire.toFixed(2) + ' · now $' + hitVal.toFixed(2);
                } else if (a.alertType === 'avwap') {
                    body = (a.condition === 'above' ? '▲ above' : '▼ below') + ' AVWAP $' + lineAtFire.toFixed(2) + ' · now $' + hitVal.toFixed(2);
                } else {
                    body = (a.condition === 'above' ? '▲ above' : '▼ below') + ' $' + Number(a.price).toFixed(2) + ' · now $' + Number(hitVal).toFixed(2);
                }
                new Notification(a.ticker + ' alert triggered', { body: body });
            }
        });
        if (toRemove.length) {
            var removeSet = {};
            toRemove.forEach(function(k) { removeSet[k] = true; });
            alertsList = alertsList.filter(function(a) { return !removeSet[_alKey(a)]; });
            _alSaveSoon();
        } else if (anyBaselineChanged) {
            // Collapses what used to be up to one kv_set write per alert
            // (fired inline inside the loop above, whenever that alert's
            // baseline reset for a new day) into at most one write per
            // alCheckTriggers() run. Multiple alerts resetting their
            // baseline in the same synchronous pass — which happens on the
            // first check of any new day, page load or not — used to fire
            // that many back-to-back kv_set calls to the same 'price_alerts'
            // key, tripping Workers KV's 1-write-per-second-per-key limit
            // and crashing the Worker before it could attach CORS headers.
            // (The arming state of trendline/AVWAP alerts rides on this same single write, and _alSaveSoon coalesces
            // it with any write a click made a moment ago.)
            _alSaveSoon();
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
        // This snaps the user to the top on every 10-second price poll AND on the async
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
                    if (a.alertType === 'trendline' || a.alertType === 'avwap') {
                        var linePctSort = _alLineAwayPct(a);
                        return linePctSort == null ? Infinity : linePctSort;
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

        var AL_PATTERN_LABELS = { inside_day: 'Inside Day', double_inside_day: 'Double Inside Day', outside_day: 'Outside Day', hammer: 'Hammer', bullish_reversal_bar: 'Bullish Reversal Bar', upside_reversal: 'Upside Reversal', oops_reversal: 'Oops Reversal', pocket_pivot: 'Pocket Pivot' };

        listEl.innerHTML = displayList.map(function(item) {
            var a = item.a, idx = item.idx;
            var key = _alKey(a);
            var fired = !!_alertFiredSess[key];
            var curr  = alertPrices[a.ticker] != null ? alertPrices[a.ticker] : null;
            var name  = (tickerMap && tickerMap[a.ticker] && tickerMap[a.ticker].name) ? tickerMap[a.ticker].name : (a.name || '');
            var condHtml = a.alertType === 'macross'
                ? (a.condition === 'above'
                    ? '<span style="color:var(--success);">▲ MA above</span>'
                    : '<span style="color:var(--danger);">▼ MA below</span>')
                : a.alertType === 'pattern'
                    ? '<span style="color:var(--text-muted-2);">⬡ Pattern</span>'
                    : a.alertType === 'trendline'
                        ? (a.condition === 'above'
                            ? '<span style="color:var(--success);">▲ trendline</span>'
                            : '<span style="color:var(--danger);">▼ trendline</span>')
                        : a.alertType === 'avwap'
                            ? (a.condition === 'above'
                                ? '<span style="color:var(--success);">▲ AVWAP</span>'
                                : '<span style="color:var(--danger);">▼ AVWAP</span>')
                        : a.isPrevDay === 'high'
                        ? '<span style="color:var(--success);">▲ PDH</span>'
                        : a.isPrevDay === 'low'
                            ? '<span style="color:var(--danger);">▼ PDL</span>'
                            : a.isPrevDay === '52wk-high'
                                ? '<span style="color:var(--success);">▲ 52WH</span>'
                                : a.isPrevDay === '52wk-low'
                                    ? '<span style="color:var(--danger);">▼ 52WL</span>'
                                    : a.condition === 'above'
                                ? '<span style="color:var(--success);">▲ above</span>'
                                : '<span style="color:var(--danger);">▼ below</span>';
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
            } else if (a.alertType === 'trendline' || a.alertType === 'avwap') {
                var linePct = fired ? null : _alLineAwayPct(a);
                if (linePct == null) {
                    awayHtml = '<div class="al-col-away">—</div>';
                } else {
                    var lineCls = linePct < 1 ? ' imminent' : linePct < 5 ? ' close' : '';
                    awayHtml = '<div class="al-col-away' + lineCls + '">' + linePct.toFixed(1) + '%</div>';
                }
            } else if (fired || curr == null) {
                awayHtml = '<div class="al-col-away">—</div>';
            } else {
                var pct = Math.abs((curr - a.price) / a.price * 100);
                var awayCls = pct < 1 ? ' imminent' : pct < 5 ? ' close' : '';
                awayHtml = '<div class="al-col-away' + awayCls + '">' + pct.toFixed(1) + '%</div>';
            }
            var prevClose = alertPrevClose[a.ticker] || null;
            var chgAbs    = (wlIsMarketOpen() && curr != null && prevClose && prevClose > 0) ? curr - prevClose : null;
            var chgPct    = (chgAbs != null) ? (chgAbs / prevClose * 100) : null;
            // Outside market hours (or no live prevClose), fall back to the
            // ticker's real last-session change instead of a same-price,
            // always-zero delta against the snapshot's own baseline price.
            if (chgPct == null) {
                var alChgSd = null;
                if (snapshot && snapshot.by_industry) {
                    outerAlChg: for (var indAlChg in snapshot.by_industry) {
                        var stAlChg = snapshot.by_industry[indAlChg];
                        for (var siAlChg = 0; siAlChg < stAlChg.length; siAlChg++) {
                            if (stAlChg[siAlChg].ticker === a.ticker) { alChgSd = stAlChg[siAlChg]; break outerAlChg; }
                        }
                    }
                }
                if (alChgSd && alChgSd.daily != null) {
                    chgPct = alChgSd.daily;
                    chgAbs = alChgSd.price ? (alChgSd.price / (1 + alChgSd.daily / 100)) * (alChgSd.daily / 100) : null;
                }
            }
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
                '<div class="al-col-target ' + a.condition + ((a.alertType === 'macross' || a.alertType === 'ma') ? ' ma-label' : '') + '"' + (a.alertType === 'pattern' ? ' data-al-tip="' + window.alGetPatternKeys(a).map(function(k){return AL_PATTERN_LABELS[k]||k;}).join('\n') + '"' : '') + '>' + (a.alertType === 'rsi14' ? 'RSI ' + a.price : a.alertType === 'macross' ? a.ma1Key.replace(/([A-Z]+)(\d+)/,'$1 $2') + ' × ' + a.ma2Key.replace(/([A-Z]+)(\d+)/,'$1 $2') : a.alertType === 'ma' ? a.maKey.replace(/([A-Z]+)(\d+)/,'$1 $2') : a.alertType === 'pattern' ? (function(){ var pk = window.alGetPatternKeys(a); var tf = (a.patternTf||'d').toUpperCase(); if (pk.length === 1) { return '<span class="al-pat-single"><span>' + (AL_PATTERN_LABELS[pk[0]]||pk[0]) + '</span><span class="al-pat-single-tf">' + tf + '</span></span>'; } return '<span class="al-pat-multi"><svg width="16" height="12" viewBox="0 0 16 12" fill="none" style="flex-shrink:0"><polyline points="0,9 3,9 5,3 7,10 9,6 11,7 13,4 16,4" stroke="var(--purple)" stroke-width="1.5" fill="none" stroke-linejoin="round" stroke-linecap="round"/></svg><span class="al-pat-multi-count">' + pk.length + '</span><span class="al-pat-multi-tf">' + tf + '</span></span>'; })() : a.alertType === 'trendline' ? '<span style="display:inline-flex;align-items:center;"><svg width="16" height="12" viewBox="0 0 16 12" fill="none" style="flex-shrink:0"><line x1="1" y1="11" x2="15" y2="1" stroke="var(--text-muted-2)" stroke-width="1.5" stroke-linecap="round"/><circle cx="2" cy="10.5" r="1.8" fill="var(--text-muted-2)"/><circle cx="14" cy="1.5" r="1.8" fill="var(--text-muted-2)"/></svg></span>' : a.alertType === 'avwap' ? '<span style="display:inline-flex;align-items:center;gap:2px;font-size:10px;color:var(--text-muted-2);letter-spacing:.5px;">AVWAP</span>' : '$' + a.price.toFixed(2)) + '</div>' +
                '<div class="al-col-curr">' + (a.alertType === 'rsi14' ? (function() { var r = null; if (snapshot && snapshot.by_industry) { outer3: for (var i3 in snapshot.by_industry) { var s3 = snapshot.by_industry[i3]; for (var j3=0;j3<s3.length;j3++) { if(s3[j3].ticker===a.ticker){r=s3[j3].rsi14;break outer3;} } } } return r != null ? r.toFixed(1) : '—'; })() : (curr != null ? '$' + curr.toFixed(2) : '—')) + '</div>' +
                chgHtml +
                chgPctHtml +
                awayHtml +
                '<div class="al-col-status">' + _alStatusPillHtml(fired, key) + '</div>' +
                '<div class="al-col-added">' + addedHtml + '</div>' +
                (a.alertType === 'trendline' || a.alertType === 'avwap'
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
            var AL_HIST_PAT_LABELS = { inside_day: 'Inside Day', double_inside_day: 'Double Inside Day', outside_day: 'Outside Day', hammer: 'Hammer', bullish_reversal_bar: 'Bullish Reversal Bar', upside_reversal: 'Upside Reversal', oops_reversal: 'Oops Reversal', pocket_pivot: 'Pocket Pivot' };
            var condHtml = f.alertType === 'macross'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? 'var(--success)' : 'var(--danger)') + ';">' +
                  (f.condition === 'above' ? '▲' : '▼') + ' ' +
                  (f.ma1Key || '').replace(/([A-Z]+)(\d+)/,'$1 $2') + ' × ' +
                  (f.ma2Key || '').replace(/([A-Z]+)(\d+)/,'$1 $2') + '</span>'
                : f.alertType === 'ma'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? 'var(--success)' : 'var(--danger)') + ';">' + (f.condition === 'above' ? '▲' : '▼') + ' ' + (f.maKey || '').replace(/([A-Z]+)(\d+)/,'$1 $2') + '</span>'
                : f.alertType === 'trendline'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? 'var(--success)' : 'var(--danger)') + ';">' + (f.condition === 'above' ? '▲' : '▼') + ' trendline' + (typeof f.lineValue === 'number' ? ' $' + f.lineValue.toFixed(2) : '') + '</span>'
                : f.alertType === 'avwap'
                ? '<span class="al-hist-cond" style="color:' + (f.condition === 'above' ? 'var(--success)' : 'var(--danger)') + ';">' + (f.condition === 'above' ? '▲' : '▼') + ' AVWAP' + (typeof f.lineValue === 'number' ? ' $' + f.lineValue.toFixed(2) : '') + '</span>'
                : f.alertType === 'pattern'
                ? (function() {
                    var pKeys = (f.triggeredPatternKeys && f.triggeredPatternKeys.length)
                        ? f.triggeredPatternKeys
                        : (f.patternKeys && f.patternKeys.length ? f.patternKeys : (f.patternKey ? [f.patternKey] : ['?']));
                    var tf = (f.patternTf||'d').toUpperCase();
                    var tipText = pKeys.map(function(k){ return AL_HIST_PAT_LABELS[k] || k.replace(/_/g,' '); }).join('\n');
                    if (pKeys.length === 1) {
                        return '<span class="al-hist-cond al-pat-single" style="color:var(--purple);" data-al-tip="' + tipText + '">' + (AL_HIST_PAT_LABELS[pKeys[0]] || pKeys[0].replace(/_/g,' ')) + ' <span class="al-pat-single-tf">' + tf + '</span></span>';
                    }
                    return '<span class="al-hist-cond al-pat-multi" data-al-tip="' + tipText + '"><svg width="16" height="12" viewBox="0 0 16 12" fill="none" style="flex-shrink:0;vertical-align:middle;margin-right:2px"><polyline points="0,9 3,9 5,3 7,10 9,6 11,7 13,4 16,4" stroke="var(--purple)" stroke-width="1.5" fill="none" stroke-linejoin="round" stroke-linecap="round"/></svg><span class="al-pat-multi-count">' + pKeys.length + '</span><span class="al-pat-multi-tf">' + tf + '</span></span>';
                })()
                : f.condition === 'above'
                    ? '<span class="al-hist-cond" style="color:var(--success);">▲ $' + (f.alertPrice||0).toFixed(2) + '</span>'
                    : '<span class="al-hist-cond" style="color:var(--danger);">▼ $' + (f.alertPrice||0).toFixed(2) + '</span>';
            var hitCls = f.condition === 'above' ? 'up' : 'dn';
            var name = esc(f.name || '');
            var curPrice = alertPrices[f.ticker];
            var chgHtml = '';
            // Line alerts have no fixed alert price; their level is the line's value when they fired
            // (older history entries without it fall back to the price they fired at).
            var sinceBase = (f.alertType === 'trendline' || f.alertType === 'avwap')
                ? (typeof f.lineValue === 'number' ? f.lineValue : f.hitPrice)
                : f.alertPrice;
            if (curPrice != null && sinceBase > 0 && f.alertType !== 'ma') {
                var chgPct = ((curPrice - sinceBase) / sinceBase) * 100;
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
        var removed = alertFiredList[idx];
        alertFiredList.splice(idx, 1);
        // The "Fired" pill on the main table and this history entry were only
        // ever kept in sync at the moment of firing — deleting the history
        // entry without this would leave the alert stuck showing "Fired"
        // with nothing backing it.
        if (removed) delete _alertFiredSess[_alFiredHistKey(removed)];
        alSaveFired();
        alUpdateBadge();
        renderHistory();
        if (currentView === 'alerts') renderAlerts();
    };

    window.alHistClearAll = function() {
        alertFiredList.forEach(function(f) { delete _alertFiredSess[_alFiredHistKey(f)]; });
        alertFiredList = [];
        alSaveFired();
        alUpdateBadge();
        renderHistory();
        if (currentView === 'alerts') renderAlerts();
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
            if (grid) grid.style.outline = '1px solid var(--danger)';
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

        var alKey = _alKey;   // one canonical key builder for every alert type (this local copy had drifted)

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
            // Was building the plain-price key for trendline/AVWAP alerts, so their real key was never cleared.
            var k = _alKey(a);
            delete _alertFiredSess[k];
            delete _alEvalStatus[k];
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
    var _alLiveTimer          = null; // drives the live-tick candle update, mirrors _mcFsLiveTimer
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

    // Re-theme the LW chart live if it's open when the toggle is flipped —
    // otherwise it'd only pick up the new theme the next time it's reopened.
    window.addEventListener('themechange', function() {
        if (!_alChart || !_alCandle) return;
        try {
            _alChart.applyOptions({
                layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted'), panes: { separatorColor: themeColor('bg-subtle'), separatorHoverColor: themeColor('bg-surface-alpha') } },
                rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted') },
                timeScale: { borderColor: themeColor('bg-surface') },
            });
            _alCandle.applyOptions({
                upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'),
                wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
            });
            if (_alVol) {
                _alVol.applyOptions({ color: themeColor('al-chart-volume') });
                _alVol.priceScale().applyOptions({ borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted') });
                if (_alOhlcv && _alOhlcv.length) {
                    _alVol.setData(_alOhlcv.map(function(d) {
                        return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
                    }));
                }
            }
        } catch (e) {}
    });

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

    var _alTooltipEnabled = false;
    var _alVolSmaMap      = null;

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
    function _addAlTrendline(p1, p2, extend, dotted) {
        return _addTrendlineCore(p1, p2, _alChart, _alCandle, _alOhlcv, _alTrendlines, { extend: !!extend, dotted: !!dotted });
    }

    // Draws an AVWAP line anchored at a bar index of the alerts chart (skips one already drawn at that anchor).
    function _addAlVwap(anchorIdx) {
        if (!_alChart || !_alOhlcv.length) return;
        if (_alVwapSeries.some(function(v) { return v.anchor === anchorIdx; })) return;
        var avData = _calcAVWAP(_alOhlcv, anchorIdx);
        if (!avData || !avData.length) return;
        var s = _alChart.addSeries(LightweightCharts.LineSeries, {
            color: _AVWAP_COLOR, lineWidth: 1.5, priceLineVisible: false,
            lastValueVisible: true, crosshairMarkerVisible: true,
        });
        s.setData(avData);
        _alVwapSeries.push({ series: s, anchor: anchorIdx, color: _AVWAP_COLOR, dataMap: new Map(avData.map(function(d) { return [d.time, d.value]; })) });
    }

    function _alTrendlineHitTest(clientX, clientY) {
        return _trendlineHitTestCore(clientX, clientY, _alChart, _alCandle, _alOhlcv, _alTrendlines, _alTrendContRef);
    }

    function _alDeselectAllTrendlines() {
        _deselectAllTrendlinesCore(_alTrendlines);
        _alSelectedTrendlineIdx = -1;
    }

    function _alSelectVwap(idx) {
        _selectVwapCore(_alVwapSeries, idx);
        _alSelectedVwapIdx = idx;
    }
    function _alDeselectAllVwaps() {
        _deselectAllVwapsCore(_alVwapSeries);
        _alSelectedVwapIdx = -1;
    }

    function _alVwapHitTest(clientX, clientY) {
        return _vwapHitTestCore(clientX, clientY, _alChart, _alVwapSeries, _alLastCrosshairTime, 'al-chart-widget');
    }

    function _alAnchorHitTest(clientX, clientY, tlIdx) {
        return _anchorHitTestCore(clientX, clientY, tlIdx, _alTrendlines, _alChart, _alCandle, _alOhlcv, _alTrendContRef);
    }

    // ── Anchor drag ───────────────────────────────────────────────────────
    function _onAlTrendAnchorDragMove(evt) {
        _onTrendAnchorDragMoveCore(evt, {
            dragState:  _alTrendDragState,
            chart:      _alChart,
            candle:     _alCandle,
            contRef:    _alTrendContRef,
            trendlines: _alTrendlines,
            ohlcv:      _alOhlcv,
            svgOverlay: _alTrendSvgOverlay,
            svgLine:    _alTrendSvgLine
        });
    }

    function _onAlTrendAnchorDragEnd() {
        _onTrendAnchorDragEndCore({
            getDragState: function() { return _alTrendDragState; },
            setDragState: function(v) { _alTrendDragState = v; },
            getSym:       function() { return _alSym; },
            trendlines:   _alTrendlines,
            contRef:      _alTrendContRef,
            svgOverlay:   _alTrendSvgOverlay,
            moveHandler:  _onAlTrendAnchorDragMove,
            endHandler:   _onAlTrendAnchorDragEnd
        });
    }

    // ── Trendline Alert creation API ──────────────────────────────────────
    // Returns { alert, created } -- `created` is false when an identical alert already existed (nothing is added).
    window.alAddTrendlineAlert = function(ticker, p1, p2, condition) {
        // Convert LW time to unix seconds (handles number, 'YYYY-MM-DD' string, or {year,month,day})
        var u1 = _alToUnix(p1 && p1.time), u2 = _alToUnix(p2 && p2.time);
        if (u1 == null || u2 == null || p1.price == null || p2.price == null) return null;
        // Normalise so np1.unix <= np2.unix (chronological order)
        var np1, np2;
        if (u1 <= u2) {
            np1 = { time: p1.time, price: p1.price, unix: u1 };
            np2 = { time: p2.time, price: p2.price, unix: u2 };
        } else {
            np1 = { time: p2.time, price: p2.price, unix: u2 };
            np2 = { time: p1.time, price: p1.price, unix: u1 };
        }
        // Dedup on the canonical key: same ticker + same two anchors (time AND price) + same condition. Price is part
        // of the key because two different lines can share both anchor dates.
        var newAlert = {
            ticker:    ticker,
            alertType: 'trendline',
            condition: condition,
            p1:        np1,
            p2:        np2,
            name:      (tickerMap && tickerMap[ticker] && tickerMap[ticker].name) ? tickerMap[ticker].name : '',
            armed:     false,   // fires on a CROSS: becomes armed once price is seen on the far side of the line
            addedAt:   new Date().toISOString()
        };
        var newKey = _alKey(newAlert);
        var existing = alertsList.filter(function(a) { return a.alertType === 'trendline' && _alKey(a) === newKey; })[0];
        if (existing) return { alert: existing, created: false };
        alertsList.push(newAlert);
        // A previous trendline alert with this exact key may have fired and been removed earlier this
        // session — that leaves _alertFiredSess holding the key, which would silently block this brand-new
        // alert from ever triggering. Clear it so the new alert starts fresh.
        delete _alertFiredSess[newKey];
        _alTryArm(newAlert);
        _alLoaded = true;
        alSave();
        alStartBackgroundPolling();
        if (currentView === 'alerts') renderAlerts();
        alStampBadges();
        return { alert: newAlert, created: true };
    };

    // Accessor for other chart contexts (fullscreen, watchlist) to read trendline alerts
    window.alGetTrendlineAlerts = function(ticker) {
        return alertsList.filter(function(a) {
            return a.alertType === 'trendline' && a.ticker === ticker && a.p1 && a.p2;
        });
    };

    // Returns { alert, created } -- `created` is false when an identical alert already existed (nothing is added).
    window.alAddAvwapAlert = function(ticker, anchorTime, condition) {
        if (!anchorTime) return null;
        var anchorUnix = _alToUnix(anchorTime);
        if (anchorUnix == null) return null;
        var newAlert = {
            ticker:     ticker,
            alertType:  'avwap',
            condition:  condition,
            anchorTime: anchorTime,
            anchorUnix: anchorUnix,
            name:       (tickerMap && tickerMap[ticker] && tickerMap[ticker].name) ? tickerMap[ticker].name : '',
            armed:      false,   // fires on a CROSS: becomes armed once price is seen on the far side of the line
            addedAt:    new Date().toISOString()
        };
        var newKey = _alKey(newAlert);
        // Dedup: same ticker + same anchor + same condition
        var existing = alertsList.filter(function(a) { return a.alertType === 'avwap' && _alKey(a) === newKey; })[0];
        if (existing) return { alert: existing, created: false };
        alertsList.push(newAlert);
        // Same reasoning as alAddTrendlineAlert: clear a stale fired-this-session key so this new alert can trigger.
        delete _alertFiredSess[newKey];
        _alTryArm(newAlert);
        _alLoaded = true;
        alSave();
        alStartBackgroundPolling();
        if (currentView === 'alerts') renderAlerts();
        alStampBadges();
        return { alert: newAlert, created: true };
    };

    // Accessor for other chart contexts (fullscreen, watchlist) to read AVWAP alerts
    window.alGetAvwapAlerts = function(ticker) {
        return alertsList.filter(function(a) { return a.alertType === 'avwap' && a.ticker === ticker; });
    };

    // ── Keeping a drawn line and its alert in step ─────────────────────────────────────────────
    // Dragging an anchor used to move only the DRAWING: the alert kept monitoring the old line, and the old line
    // was redrawn from the alert store the next time the chart opened. Deleting a line with the Delete key left
    // its alert armed as well. The two APIs below close both gaps; the chart code in multichart.js calls them.

    // A drawn trendline's anchors were dragged: move any alert matching the OLD points to the new ones.
    // Matches on anchor time AND price (two different lines can share both dates). The alert is re-baselined so
    // dragging a line through the current price can't fire it. Returns how many alerts moved.
    window.alSyncTrendlineAlertsAfterDrag = function(ticker, oldP1, oldP2, newP1, newP2) {
        var ou1 = _alPtUnix(oldP1), ou2 = _alPtUnix(oldP2);
        var nu1 = _alToUnix(newP1 && newP1.time), nu2 = _alToUnix(newP2 && newP2.time);
        if (ou1 == null || ou2 == null || nu1 == null || nu2 == null || newP1.price == null || newP2.price == null) return 0;
        var first = nu1 <= nu2;
        var lo = first ? newP1 : newP2, hi = first ? newP2 : newP1;
        var moved = 0;
        alertsList.forEach(function(a) {
            if (a.alertType !== 'trendline' || a.ticker !== ticker || !a.p1 || !a.p2) return;
            if (_alPtUnix(a.p1) !== ou1 || _alPtUnix(a.p2) !== ou2) return;
            if (oldP1.price != null && a.p1.price !== oldP1.price) return;
            if (oldP2.price != null && a.p2.price !== oldP2.price) return;
            var oldKey = _alKey(a);
            a.p1 = { time: lo.time, price: lo.price, unix: _alToUnix(lo.time) };
            a.p2 = { time: hi.time, price: hi.price, unix: _alToUnix(hi.time) };
            a.armed = false;
            _alTryArm(a);
            delete _alEvalStatus[oldKey];
            delete _alertFiredSess[_alKey(a)];
            moved++;
        });
        if (!moved) return 0;
        // If the moved line now coincides exactly with another alert's line, keep one of them.
        var seen = {};
        alertsList = alertsList.filter(function(a) {
            if (a.alertType !== 'trendline') return true;
            var k = _alKey(a);
            if (seen[k]) return false;
            seen[k] = true;
            return true;
        });
        _alLoaded = true;
        alSave();
        if (currentView === 'alerts') renderAlerts();
        alStampBadges();
        return moved;
    };

    // Alerts backed by a drawn line.
    //   kind 'trendline': ref = the drawing's anchor points { l: {time, price}, r: {time, price} }
    //   kind 'avwap':     ref = { ohlcv, tf, idx } -- the chart's bar array, its timeframe, and the anchor's index in it
    window.alLineAlerts = function(ticker, kind, ref) {
        if (kind === 'trendline') {
            var u1 = _alPtUnix(ref.l), u2 = _alPtUnix(ref.r);
            return alertsList.filter(function(a) {
                return a.alertType === 'trendline' && a.ticker === ticker && a.p1 && a.p2 &&
                       _alPtUnix(a.p1) === u1 && _alPtUnix(a.p2) === u2 &&
                       (ref.l.price == null || a.p1.price === ref.l.price) &&
                       (ref.r.price == null || a.p2.price === ref.r.price);
            });
        }
        return alertsList.filter(function(a) {
            return a.alertType === 'avwap' && a.ticker === ticker &&
                   _chartAnchorIdx(ref.ohlcv, ref.tf, _alAnchorUnix(a)) === ref.idx;
        });
    };
    // Delete the alert(s) behind a drawn line, asking first (unless "don't ask again" was ticked this session),
    // then run proceed() -- which removes the drawing itself. With no alert behind the line it just runs proceed().
    window.alDeleteLineAlerts = function(ticker, kind, ref, proceed) {
        var matches = window.alLineAlerts(ticker, kind, ref);
        if (!matches.length) { proceed(); return; }
        var doIt = function() {
            matches.map(function(a) { return alertsList.indexOf(a); })
                   .filter(function(i) { return i >= 0; })
                   .sort(function(x, y) { return y - x; })          // from the end, so earlier indices stay valid
                   .forEach(function(i) { window.alDelete(i); });
            alStampBadges();
            if (currentView === 'alerts') renderAlerts();
            proceed();
        };
        if (_alSkipDeleteConfirm) { doIt(); return; }
        var n = matches.length;
        alConfirmOpen(
            'Delete line and alert' + (n > 1 ? 's' : '') + '?',
            'This line has ' + n + ' active alert' + (n > 1 ? 's' : '') + ' on ' + ticker + '. Deleting the line also deletes ' + (n > 1 ? 'them' : 'the alert') + '.',
            doIt, 'Delete', true
        );
    };

    // ── AL Measure drag handlers ─────────────────────────────────────────────
    function _onAlMeasureDragMove(evt) {
        _onMeasureDragMoveCore(evt, {
            getActive:  function() { return _alMeasureActive; },
            contRef:    _alTrendContRef,
            chart:      _alChart,
            candle:     _alCandle,
            ohlcv:      _alOhlcv,
            getStart:   function() { return _alMeasureStart; },
            getRafId:   function() { return _alMeasureRafId; },
            setRafId:   function(v) { _alMeasureRafId = v; },
            setResult:  function(v) { _alMeasureResult = v; },
            svgOverlay: _alMeasureSvgOverlay,
            svgRect:    _alMeasureSvgRect,
            hLine:      _alMeasureHLine,
            infoDiv:    _alMeasureInfoDiv
        });
    }
    function _onAlMeasureDragEnd() {
        _onMeasureDragEndCore({
            moveHandler: _onAlMeasureDragMove,
            endHandler:  _onAlMeasureDragEnd,
            setActive:   function(v) { _alMeasureActive = v; }
        });
    }
    function _onAlMeasurePreviewMove(evt) {
        _onMeasurePreviewMoveCore(evt, {
            getActive:  function() { return _alMeasureActive; },
            getPhase:   function() { return _alMeasurePhase; },
            contRef:    _alTrendContRef,
            chart:      _alChart,
            candle:     _alCandle,
            ohlcv:      _alOhlcv,
            getStart:   function() { return _alMeasureStart; },
            getRafId:   function() { return _alMeasureRafId; },
            setRafId:   function(v) { _alMeasureRafId = v; },
            setResult:  function(v) { _alMeasureResult = v; },
            svgOverlay: _alMeasureSvgOverlay,
            svgRect:    _alMeasureSvgRect,
            hLine:      _alMeasureHLine,
            infoDiv:    _alMeasureInfoDiv
        });
    }

    function _onAlTrendMouseDown(evt) {
        _onTrendMouseDownCore(evt, {
            candle:  _alCandle,
            chart:   _alChart,
            contRef: _alTrendContRef,
            getMeasureMode:    function() { return _alMeasureMode; },
            getDragState:      function() { return _alTrendDragState; },
            setDragState:      function(v) { _alTrendDragState = v; },
            trendDraw:         _alTrendDraw,
            svgOverlay:        _alTrendSvgOverlay,
            svgLine:           _alTrendSvgLine,
            ohlcv:             _alOhlcv,
            getMeasurePhase:   function() { return _alMeasurePhase; },
            setMeasurePhase:   function(v) { _alMeasurePhase = v; },
            getMeasureResult:  function() { return _alMeasureResult; },
            setMeasureResult:  function(v) { _alMeasureResult = v; },
            setMeasureActive:  function(v) { _alMeasureActive = v; },
            getMeasureRafId:   function() { return _alMeasureRafId; },
            setMeasureRafId:   function(v) { _alMeasureRafId = v; },
            getMeasureStart:   function() { return _alMeasureStart; },
            setMeasureStart:   function(v) { _alMeasureStart = v; },
            measureSvgOverlay: _alMeasureSvgOverlay,
            measureSvgRect:    _alMeasureSvgRect,
            measureHLine:      _alMeasureHLine,
            measureInfoDiv:    _alMeasureInfoDiv,
            measurePreviewMoveHandler: _onAlMeasurePreviewMove,
            getSelectedIdx:    function() { return _alSelectedTrendlineIdx; },
            setSelectedIdx:    function(v) { _alSelectedTrendlineIdx = v; },
            trendlines:        _alTrendlines,
            deselectAllTrendlines: _alDeselectAllTrendlines,
            deselectAllVwaps:  _alDeselectAllVwaps,
            anchorHitTest:     _alAnchorHitTest,
            trendlineHitTest:  _alTrendlineHitTest,
            dragMoveHandler:   _onAlTrendAnchorDragMove,
            dragEndHandler:    _onAlTrendAnchorDragEnd,
            getTrendlineMode:  function() { return _alTrendlineMode; },
            setTrendlineMode:  function(v) { _alTrendlineMode = v; },
            getTrendlineStyle: function() { return _alTlMenu.getStyle(); },
            getLastCrosshairTime: function() { return _alLastCrosshairTime; },
            addTrendline:      _addAlTrendline,
            doneBtnId:         'al-chart-trendline-btn'
        });
    }

    function _onAlTrendMouseMove(evt) {
        _onTrendMouseMoveCore(evt, {
            getTrendDraw:    function() { return _alTrendDraw; },
            svgOverlay:      _alTrendSvgOverlay,
            svgLine:         _alTrendSvgLine,
            candle:          _alCandle,
            chart:           _alChart,
            contRef:         _alTrendContRef,
            trendlines:      _alTrendlines,
            getTrendlineMode: function() { return _alTrendlineMode; },
            getDragState:    function() { return _alTrendDragState; },
            getSelectedIdx:  function() { return _alSelectedTrendlineIdx; },
            anchorHitTest:   _alAnchorHitTest,
            trendlineHitTest: _alTrendlineHitTest
        });
    }

    // ── Right-click context menu ──────────────────────────────────────────
    var _alCtxTrendline = null; // {p1, p2} when right-click lands on a trendline
    var _alCtxAvwap     = null; // {anchorIdx, anchorTime} when right-click lands on an AVWAP line

    // (trendline/AVWAP value functions live with the rest of the line engine, next to alFetchPrices)

    function _alDismissCtx() {
        _hideCtxMenu('al-chart-ctx-menu');
        _alCtxPrice     = null;
        _alCtxMa        = null;
        _alCtxTrendline = null;
        _alCtxAvwap     = null;
    }

    window.alChartCtxAlert = function(direction) {
        _ctxAlertCore(direction, {
            getCtxTrendline: function() { return _alCtxTrendline; },
            getCtxAvwap:     function() { return _alCtxAvwap; },
            getCtxMa:        function() { return _alCtxMa; },
            getCtxPrice:     function() { return _alCtxPrice; },
            getSym:          function() { return _alSym; },
            dismiss:         _alDismissCtx
        });
    };

    function _alAttachCtxMenu() {
        _attachCtxMenuCore({
            getAttached: function() { return _alCtxAttached; },
            setAttached: function(v) { _alCtxAttached = v; },
            parentElId: 'al-chart-widget-wrap',
            chartDivId: 'al-chart-widget',
            getTooltipEnabled: function() { return _alTooltipEnabled; },
            setTooltipEnabled: function(v) { _alTooltipEnabled = v; },
            tooltipBtnId: 'al-chart-tooltip-btn',
            getMeasurePhase:  function() { return _alMeasurePhase; },
            setMeasurePhase:  function(v) { _alMeasurePhase = v; },
            setMeasureActive: function(v) { _alMeasureActive = v; },
            getMeasureRafId:  function() { return _alMeasureRafId; },
            setMeasureRafId:  function(v) { _alMeasureRafId = v; },
            measurePreviewMoveHandler: _onAlMeasurePreviewMove,
            getMeasureSvgOverlay: function() { return _alMeasureSvgOverlay; },
            getMeasureInfoDiv:    function() { return _alMeasureInfoDiv; },
            getMeasureResult: function() { return _alMeasureResult; },
            setMeasureResult: function(v) { _alMeasureResult = v; },
            trendDraw:     _alTrendDraw,
            getSvgOverlay: function() { return _alTrendSvgOverlay; },
            getVwapMode: function() { return _alVwapMode; },
            setVwapMode: function(v) { _alVwapMode = v; },
            vwapBtnId: 'al-chart-vwap-btn',
            getChart: function() { return _alChart; },
            getSym:   function() { return _alSym; },
            trendlineHitTest: _alTrendlineHitTest,
            getTrendlines: function() { return _alTrendlines; },
            ctxAboveTxtId:  'al-chart-ctx-above-txt',
            ctxBelowTxtId:  'al-chart-ctx-below-txt',
            ctxMenuId:      'al-chart-ctx-menu',
            setCtxTrendline: function(v) { _alCtxTrendline = v; },
            setCtxPrice:     function(v) { _alCtxPrice = v; },
            setCtxMa:        function(v) { _alCtxMa = v; },
            vwapHitTest: _alVwapHitTest,
            getVwapSeries: function() { return _alVwapSeries; },
            getOhlcv:      function() { return _alOhlcv; },
            setCtxAvwap: function(v) { _alCtxAvwap = v; },
            getCandle: function() { return _alCandle; },
            getLastCrosshairPrice: function() { return _alLastCrosshairPrice; },
            getLastCrosshairTime:  function() { return _alLastCrosshairTime; },
            getMaDataMap: function() { return _alMaDataMap; },
            getMaSeries:  function() { return _alMaSeries; },
            dismissCtx: _alDismissCtx
        });
    }

    // ── Core chart destroy / build ────────────────────────────────────────
    function _destroyAlChart() {
        _alStopLiveTick();
        if (_alChart) { try { _alChart.remove(); } catch(e) {} _alChart = null; }
        _alCandle = null; _alVol = null; _alVolMa = null; _alVolData = null; _alVolSmaMap = null;
        _alMaSeries = {}; _alMaDataMap = {};
        _alVwapSeries = []; _alTrendlines = []; _alTrendlineFirst = null;
        _alTrendSvgOverlay = null; _alTrendSvgLine = null;
        _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
        _alTrendlineMode = false; _alSelectedTrendlineIdx = -1; _alSelectedVwapIdx = -1;
        _alLastCrosshairTime = null; _alSym = null;
        _alTooltipEnabled = false;
        if (_alKeyHandler) { document.removeEventListener('keydown', _alKeyHandler); _alKeyHandler = null; }
        var mktEl = document.getElementById('al-chart-mkt-info');
        if (mktEl) mktEl.style.display = 'none';
        var tBtn  = document.getElementById('al-chart-trendline-btn');
        if (tBtn)  tBtn.classList.remove('active');
        var vBtn  = document.getElementById('al-chart-vwap-btn');
        if (vBtn)  vBtn.classList.remove('active');
        var ttBtn = document.getElementById('al-chart-tooltip-btn');
        if (ttBtn) ttBtn.classList.remove('active');
        if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
        var maPanel   = document.getElementById('al-chart-ma-panel');
        var maChevron = document.getElementById('al-chart-ma-chevron');
        if (maPanel)   maPanel.style.display = 'none';
        if (maChevron) maChevron.style.transform = '';
    }

    // ── Alerts chart live tick ───────────────────────────────────────────
    // alFetchPrices() already runs every 10s during market hours, but only
    // for trigger-checking and the AVWAP cache — it never touches the
    // visible chart candle. That gap is why an open alerts chart sat frozen
    // at whatever price it had on open while the fullscreen chart (which has
    // its own _mcFsStartLiveTick) kept updating. This is a direct mirror of
    // that function, targeting _alCandle/_alOhlcv/_alSym instead of the
    // _mcFs* equivalents — an independent fetch rather than reusing
    // alFetchPrices()'s data, since this chart can be opened for a ticker
    // that isn't necessarily in alertsList/alertFiredList yet (e.g. while
    // still building a new alert), so alertPrices[ticker] isn't guaranteed
    // to exist.
    function _alStartLiveTick(sym, tf) {
        _alStopLiveTick();
        if (!wlIsMarketOpen()) return;
        _alLiveTimer = setInterval(function() {
            if (_alSym !== sym || !_alCandle) { _alStopLiveTick(); return; }
            if (!wlIsMarketOpen()) { _alStopLiveTick(); return; }
            fetch(WL_PROXY + '?action=quotes_batch&tickers=' + encodeURIComponent(sym))
                .then(function(r) { return r.ok ? r.json() : null; })
                .then(function(data) {
                    var q = data && data.quotes && data.quotes[0];
                    if (!q || !q.price || _alSym !== sym || !_alCandle || !_alOhlcv.length) return;
                    if (tf !== 'D') {
                        // W/M: fold into the current period's bar (helper lives in multichart.js)
                        var _wm = _mcApplyLiveWM(_alOhlcv, tf, q.price, q.dayHigh, q.dayLow, false);
                        if (_wm) { try { _alCandle.update(_wm); } catch(e) {} }
                        return;
                    }
                    var now = new Date();
                    var todayTs = Math.floor(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 1000) + 43200;
                    var last = _alOhlcv[_alOhlcv.length - 1];
                    var lastDayTs = Math.floor(last.time / 86400) * 86400 + 43200;
                    if (lastDayTs !== todayTs) return; // new trading day — let the next open/reload pick it up
                    var high = q.dayHigh != null ? Math.max(last.high, q.dayHigh, q.price) : Math.max(last.high, q.price);
                    var low  = q.dayLow  != null ? Math.min(last.low,  q.dayLow,  q.price) : Math.min(last.low,  q.price);
                    last.high = high; last.low = low; last.close = q.price;
                    try { _alCandle.update({ time: todayTs, open: last.open, high: high, low: low, close: q.price, volume: last.volume }); } catch(e) {}
                }).catch(function() {});
        }, 10 * 1000);
    }

    function _alStopLiveTick() {
        if (_alLiveTimer) { clearInterval(_alLiveTimer); _alLiveTimer = null; }
    }

    function _alRenderPrePostBadge(sym) {
        // Shared core lives in multichart.js — same cross-file pattern this
        // codebase already uses for fetchMcOhlcv/fetchMcPrePost.
        _renderPrePostBadge({
            sym: sym,
            badgeId: 'al-chart-prepost-badge',
            getChart: function() { return _alChart; },
            getCurrentSym: function() { return _alSym; },
            isOpen: function() {
                var panel = document.getElementById('al-chart-panel');
                return !!panel && panel.classList.contains('open');
            }
        });
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
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;">No data</div>';
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
            _alTrendSvgLine.setAttribute('stroke', _TRENDLINE_COLOR());
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
            layout: { background: { color: themeColor('bg-page') }, textColor: themeColor('text-muted'), panes: { separatorColor: themeColor('bg-subtle'), separatorHoverColor: themeColor('bg-surface-alpha') } },
            grid:    { vertLines: { visible: false }, horzLines: { visible: false } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted'), scaleMargins: { top: 0.05, bottom: 0.02 } },
            timeScale: { borderColor: themeColor('bg-surface'), timeVisible: false, secondsVisible: false, rightOffset: 12 },
            handleScroll: true, handleScale: true,
        });
        _alAttachCtxMenu();

        // Candle series
        _alCandle = _alChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: themeColor('al-chart-up'), downColor: themeColor('al-chart-down'), borderVisible: false,
            wickUpColor: themeColor('al-chart-up'), wickDownColor: themeColor('al-chart-down'),
            priceLineVisible: false, lastValueVisible: true,
        });
        _alCandle.setData(_alOhlcv);

        // Volume pane
        _alVol = _alChart.addSeries(LightweightCharts.HistogramSeries, {
            color: themeColor('al-chart-volume'), priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: true,
        }, 1);
        _alVol.setData(_alOhlcv.map(function(d) {
            return { time: d.time, value: d.volume, color: d.close >= d.open ? themeColor('al-chart-vol-up-alpha') : themeColor('al-chart-vol-down-alpha') };
        }));
        _alVol.priceScale().applyOptions({ visible: true, borderColor: themeColor('bg-surface'), textColor: themeColor('text-muted'), minimumWidth: 60 });

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
        _alVolSmaMap = _alVolData && _alVolData.length
            ? new Map(_alVolData.map(function(d) { return [d.time, d.value]; }))
            : null;

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
            var color = volDiffPct >= 0 ? 'var(--success)' : 'var(--danger)';
            var lbl = document.createElement('div');
            lbl.id = 'al-chart-vol-pct-label';
            lbl.style.cssText = 'position:absolute;z-index:20;pointer-events:none;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:3px;white-space:nowrap;line-height:1;';
            lbl.innerHTML = '<span style="color:var(--border-muted);">›</span>'
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
                // Select the new AVWAP so Delete removes it straight away (trendlines deselected first, as on fullscreen).
                _alDeselectAllTrendlines();
                _alSelectVwap(_alVwapSeries.length - 1);
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
        leg.style.cssText = 'position:absolute;top:8px;left:14px;z-index:10;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-muted-2);pointer-events:none;line-height:1.8;background:var(--bg-page-alpha-3);padding:4px 10px;border-radius:4px;';
        container.style.position = 'relative';
        container.appendChild(leg);

        // Pre/post-market price badge — same treatment as the fullscreen/watchlist
        // charts: no background/border, right-offset computed dynamically off the
        // price-scale's actual rendered width so it never overlaps axis labels.
        var alPrepost = document.createElement('div');
        alPrepost.id = 'al-chart-prepost-badge';
        alPrepost.style.cssText = 'position:absolute;top:8px;right:8px;z-index:10;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;pointer-events:none;line-height:1.6;padding:3px 8px;border-radius:4px;display:none;';
        container.appendChild(alPrepost);
        _alRenderPrePostBadge(sym);

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
            if (!p.time || !p.seriesData || !p.seriesData.size) {
                leg.innerHTML = '';
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
                return;
            }
            var d = p.seriesData.get(_alCandle);
            if (!d) {
                leg.innerHTML = '';
                if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
                return;
            }
            var cl = d.close >= d.open ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
            var vd = p.seriesData.get(_alVol);
            var chgHtml = '';
            var barIdx = _barIdxByTime(_alOhlcv, p.time);
            if (barIdx > 0) {
                var prevClose = _alOhlcv[barIdx - 1].close;
                var delta = d.close - prevClose;
                var pct = (delta / prevClose) * 100;
                var chgClr = delta >= 0 ? 'var(--success)' : 'var(--danger)';
                chgHtml = '&nbsp;&nbsp;<span style="color:' + chgClr + '">'
                        + (delta >= 0 ? '+' : '') + delta.toFixed(2)
                        + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)'
                        + '</span>';
            }
            leg.innerHTML =
                '<span style="color:var(--text-muted-2)">O</span> <span style="color:'+cl+'">'+fp(d.open)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">H</span> <span style="color:'+cl+'">'+fp(d.high)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">L</span> <span style="color:'+cl+'">'+fp(d.low)+'</span>&nbsp; ' +
                '<span style="color:var(--text-muted-2)">C</span> <span style="color:'+cl+'">'+fp(d.close)+'</span>' +
                chgHtml +
                (vd ? '&nbsp;&nbsp;<span style="color:var(--text-muted)">Vol</span> <span style="color:var(--text-muted-2)">' + fv(vd.value) + '</span>' : '');
            // Floating tooltip
            if (_alTooltipEnabled) {
                var ttDiv = _getLwTooltipDiv();
                ttDiv.innerHTML = _buildTooltipHtml(d, barIdx, _alOhlcv, _alVolSmaMap, _alMaDataMap, _alActiveMas, p.time);
                ttDiv.style.display = 'block';
                if (p.point) {
                    var rect = container.getBoundingClientRect();
                    _positionTooltip(ttDiv, rect.left + p.point.x, rect.top + p.point.y, rect.right);
                }
            } else if (_lwTooltipDiv) {
                _lwTooltipDiv.style.display = 'none';
            }
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
            var chgColor = chg >= 0 ? 'var(--success)' : 'var(--danger)';
            var chgSign  = chg >= 0 ? '+' : '';
            var barLabel = tf === 'W' ? 'WK' : tf === 'M' ? 'MO' : 'DAY';
            var barColor = chg >= 0 ? 'var(--al-chart-up)' : 'var(--al-chart-down)';
            function mkBar(low, high, curr, width, crLabel) {
                var pos = (high > low) ? Math.max(2, Math.min(98, (curr - low) / (high - low) * 100)) : 50;
                var p = pos.toFixed(1);
                var crSpan = crLabel != null
                    ? '<span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-150%);font-size:9px;font-weight:700;color:' + crLabel.color + ';letter-spacing:.02em;pointer-events:none;">' + crLabel.text + '</span>'
                    : '';
                return '<span style="position:relative;display:inline-block;width:' + width + 'px;height:4px;border-radius:2px;background:var(--bg-surface);vertical-align:middle;flex-shrink:0;overflow:visible;">'
                    + '<span style="position:absolute;left:0;top:0;height:100%;width:' + p + '%;background:' + barColor + ';border-radius:2px;"></span>'
                    + '<span style="position:absolute;top:50%;left:' + p + '%;transform:translate(-50%,-50%);width:8px;height:8px;background:var(--text-primary-alt);border-radius:50%;box-shadow:0 0 0 1.5px var(--bg-page);"></span>'
                    + crSpan + '</span>';
            }
            var crRaw   = (dayHigh > dayLow) ? Math.round((close - dayLow) / (dayHigh - dayLow) * 100) : null;
            var crLabel = crRaw != null ? { text: crRaw + '%', color: crRaw >= 60 ? 'var(--success)' : crRaw >= 30 ? 'var(--warning-alt)' : 'var(--danger)' } : null;
            var adrEl = document.getElementById('al-chart-mkt-adr');
            if (adrEl) {
                var adrSd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var adrRaw = adrSd ? adrSd.adr_pct : null;
                if (adrRaw != null) {
                    adrEl.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">ADR%</span>'
                                    + '<span style="color:var(--text-primary-alt);font-size:12px;">' + adrRaw.toFixed(1) + '%</span>';
                    adrEl.style.display = 'inline-flex';
                } else { adrEl.style.display = 'none'; }
            }
            var mcapEl = document.getElementById('al-chart-mkt-mcap');
            if (mcapEl) {
                var sd = tickerMap && tickerMap[sym] ? tickerMap[sym] : null;
                var mcapRaw = sd ? sd.MarketCap : null;
                if (mcapRaw != null) {
                    var mc = mcapRaw >= 1e12 ? (mcapRaw/1e12).toFixed(2)+'T' : mcapRaw >= 1e9 ? (mcapRaw/1e9).toFixed(2)+'B' : mcapRaw >= 1e6 ? (mcapRaw/1e6).toFixed(0)+'M' : mcapRaw;
                    mcapEl.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">Mkt Cap</span><span style="color:var(--text-primary-alt);font-size:12px;">' + mc + '</span>';
                    mcapEl.style.display = 'inline-flex';
                } else { mcapEl.style.display = 'none'; }
            }
            document.getElementById('al-chart-mkt-price').innerHTML =
                '<span style="color:var(--text-emphasis-2);font-size:17px;font-weight:700;">' + fp(close) + '</span>' +
                '&nbsp;<span style="color:' + chgColor + ';font-size:13px;font-weight:600;">' + chgSign + fp(chg) + '&nbsp;(' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)</span>';
            document.getElementById('al-chart-mkt-day').innerHTML =
                '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">' + barLabel + '</span>' +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(dayLow) + '</span>' +
                mkBar(dayLow, dayHigh, close, 130, crLabel) +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(dayHigh) + '</span>';
            var w52HiPct   = (yrHigh > 0) ? (yrHigh - close) / yrHigh * 100 : 0;
            var w52HiLabel = yrHigh > 0 ? {
                text:  w52HiPct < 0.5 ? 'ATH' : ('-' + w52HiPct.toFixed(1) + '%'),
                color: w52HiPct <= 5 ? 'var(--success)' : w52HiPct <= 15 ? 'var(--warning-alt)' : 'var(--danger)'
            } : null;
            document.getElementById('al-chart-mkt-52w').innerHTML =
                '<span style="color:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.04em;">52W</span>' +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(yrLow) + '</span>' +
                mkBar(yrLow, yrHigh, close, 120, w52HiLabel) +
                '<span style="color:var(--text-primary-alt);font-size:12px;">' + fp(yrHigh) + '</span>';
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
            // Alt shortcuts: D = tooltip, T = trendline, A = AVWAP
            if (evt.altKey && !evt.ctrlKey && !evt.metaKey) {
                if (evt.key === 'd' || evt.key === 'D') { evt.preventDefault(); window.alChartToggleTooltip(); return; }
                if (evt.key === 't' || evt.key === 'T') { evt.preventDefault(); window.alChartToggleTrendline(); return; }
                if (evt.key === 'a' || evt.key === 'A') { evt.preventDefault(); window.alChartToggleVwap(); return; }
            }
            if (evt.key !== 'Delete') return;
            // A line that backs an alert takes the alert with it (after a confirm). Before, the alert stayed armed
            // and the line was redrawn from the alert store the next time the chart opened.
            if (_alSelectedTrendlineIdx !== -1) {
                evt.stopPropagation();
                var selTl = _alTrendlines[_alSelectedTrendlineIdx];
                _alSelectedTrendlineIdx = -1;
                if (selTl) _deleteTrendlineWithAlerts(_alSym, selTl, function() {
                    var ti = _alTrendlines.indexOf(selTl);
                    if (ti !== -1) _alTrendlines.splice(ti, 1);
                    try { if (_alCandle) _alCandle.detachPrimitive(selTl.primitive); } catch(e) {}
                });
                return;
            }
            if (_alSelectedVwapIdx !== -1) {
                evt.stopPropagation();
                var selVwap = _alVwapSeries[_alSelectedVwapIdx];
                _alSelectedVwapIdx = -1;
                if (selVwap) _deleteVwapWithAlerts(_alSym, _alOhlcv, _alChartTf, selVwap, function() {
                    var vi = _alVwapSeries.indexOf(selVwap);
                    if (vi !== -1) _alVwapSeries.splice(vi, 1);
                    try { _alChart.removeSeries(selVwap.series); } catch(e) {}
                    _alVwapSeries.forEach(function(entry) { _vwapSetSelectedLook(entry, false); });
                });
                return;
            }
            if (_alTrendlineMode && _alTrendlines.length) {
                evt.stopPropagation();
                var tLast = _alTrendlines[_alTrendlines.length - 1];
                _deleteTrendlineWithAlerts(_alSym, tLast, function() {
                    var li = _alTrendlines.indexOf(tLast);
                    if (li !== -1) _alTrendlines.splice(li, 1);
                    try { if (_alCandle) _alCandle.detachPrimitive(tLast.primitive); } catch(e) {}
                });
            }
        };
        document.addEventListener('keydown', _alKeyHandler);

        // Tooltip button (injected once, idempotent)
        (function() {
            var avwapBtn = document.getElementById('al-chart-vwap-btn');
            if (avwapBtn && !document.getElementById('al-chart-tooltip-btn')) {
                var ttBtn = document.createElement('button');
                ttBtn.id        = 'al-chart-tooltip-btn';
                ttBtn.className = avwapBtn.className.replace(/\bactive\b/g, '').trim();
                ttBtn.title     = 'Data Tooltip (Alt+D)';
                ttBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><line x1="6" y1="1" x2="6" y2="11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><line x1="1" y1="6" x2="11" y2="6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
                ttBtn.addEventListener('click', window.alChartToggleTooltip);
                avwapBtn.parentNode.insertBefore(ttBtn, avwapBtn.nextSibling);
            }
            var existing = document.getElementById('al-chart-tooltip-btn');
            if (existing) existing.classList.toggle('active', _alTooltipEnabled);
        })();

        // Inject live bar
        _injectChartLiveBar(sym, tf, _alCandle, _alVol, _alOhlcv,
            function() { return _alSym !== sym || !_alCandle; });
        _alStartLiveTick(sym, tf);

        // Restore alert-backed trendlines and AVWAPs so they're visible when reviewing the chart. The anchor is
        // resolved on THIS chart's timeframe (the old strict time match drew nothing whenever the timeframe differed).
        _restoreAlertLines(sym, tf, _alOhlcv, _addAlTrendline, _addAlVwap);
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
                    var rankColor = rankPct != null ? (rankPct >= 75 ? 'var(--success)' : rankPct >= 40 ? 'var(--warning-alt)' : 'var(--danger)') : 'var(--text-muted)';
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
        _alTooltipEnabled = false;
        var ttBtn = document.getElementById('al-chart-tooltip-btn');
        if (ttBtn) ttBtn.classList.remove('active');
        if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
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

        var tf = _alChartTf;
        _alVisibleBars = tf === 'D' ? 252 : tf === 'W' ? 104 : 60;
        _alSym = ticker;
        var widgetDiv = document.getElementById('al-chart-widget');
        widgetDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;">Loading\u2026</div>';
        var loadTicker = ticker;
        fetchMcOhlcv(ticker, tf).then(function(ohlcv) {
            if (_alSym !== loadTicker || _alChartTf !== tf) return;
            _buildAlChart(loadTicker, ohlcv, tf);
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
        _alTooltipEnabled = false;
        var ttBtn = document.getElementById('al-chart-tooltip-btn');
        if (ttBtn) ttBtn.classList.remove('active');
        if (_lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
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
        var sym = _alSym;   // (no cache delete: the alert engine reads this same series; a forced fetch replaces it only on success)
        var container = document.getElementById('al-chart-widget');
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--border-muted);font-size:12px;">Loading\u2026</div>';
        fetchMcOhlcv(sym, tf, false, true).then(function(ohlcv) {
            if (_alSym !== sym || _alChartTf !== tf) return;
            _buildAlChart(sym, ohlcv || _mcOhlcvCache[sym + '_' + tf] || null, tf);
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
        _alTlMenu.resetStyle(); // plain click and Alt+T always draw solid
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

    // Alerts chart instance of the shared hold-menu (_makeTrendlineStyleHold, defined in multichart.js) — #al-chart-trendline-btn
    var _alTlMenu = _makeTrendlineStyleHold({
        toggle:     function() { window.alChartToggleTrendline(); },
        isActive:   function() { return _alTrendlineMode; },
        cancelDraw: function() {
            _alTrendDraw.active = false; _alTrendDraw.startTime = null; _alTrendDraw.startPrice = null;
            if (_alTrendSvgOverlay) _alTrendSvgOverlay.style.display = 'none';
        }
    });
    window.alChartTlBtnDown  = _alTlMenu.down;
    window.alChartTlBtnClick = _alTlMenu.click;

    window.alChartToggleTooltip = function() {
        _alTooltipEnabled = !_alTooltipEnabled;
        var btn = document.getElementById('al-chart-tooltip-btn');
        if (btn) btn.classList.toggle('active', _alTooltipEnabled);
        if (!_alTooltipEnabled && _lwTooltipDiv) _lwTooltipDiv.style.display = 'none';
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
        // Confirm an open delete dialog first, if any — lets Delete → Enter
        // chain through the list with no mouse, while still requiring an
        // explicit confirm on every single deletion.
        if (document.getElementById('al-confirm-overlay').classList.contains('open')) {
            e.preventDefault();
            alConfirmClear();
            return;
        }
        if (document.getElementById('al-modal-overlay').classList.contains('open')) alSubmitForm();
    });
    var _alConfirmCallback = null;
    var _alSkipDeleteConfirm = false; // session-only — plain in-memory flag, resets on reload/next visit. Set via the "don't ask again" checkbox, delete-only and opt-in per call site (see showSkipOption below) so it can't silently show up on some future non-delete confirmation.

    // Shared re-sync so the keyboard cursor lands on the correct row whether
    // a delete went through the confirm dialog or skipped it entirely.
    function _alResyncKbAfterDelete() {
        if (_alKbFocus !== 'history' && _alKbIdx >= 0) {
            var rows = alKbGetAlertRows();
            if (rows.length) {
                _alKbIdx = Math.min(_alKbIdx, rows.length - 1);
                alKbSetAlertActive(rows, _alKbIdx);
            } else {
                _alKbIdx = -1;
            }
        }
    }

    // Lazily creates (once) the "don't ask again" row right after the confirm
    // message, since that markup isn't in this file to add directly. Visibility
    // is toggled per-call by alConfirmOpen's showSkipOption argument.
    function _alEnsureSkipCheckboxRow() {
        var existing = document.getElementById('al-confirm-skip-row');
        if (existing) return existing;
        var msgEl = document.getElementById('al-confirm-msg');
        if (!msgEl) return null;
        var row = document.createElement('label');
        row.id = 'al-confirm-skip-row';
        row.style.cssText = 'display:flex;align-items:center;gap:6px;margin:12px 0 18px;font-size:12px;color:var(--text-muted-2);cursor:pointer;user-select:none;';
        var box = document.createElement('input');
        box.type = 'checkbox';
        box.id = 'al-confirm-skip-checkbox';
        box.style.cssText = 'margin:0;cursor:pointer;';
        var span = document.createElement('span');
        span.textContent = "Don't ask again this session";
        row.appendChild(box);
        row.appendChild(span);
        msgEl.insertAdjacentElement('afterend', row);
        return row;
    }

    window.alConfirmOpen = function(title, msg, callback, okLabel, showSkipOption) {
        document.getElementById('al-confirm-title').textContent = title;
        document.getElementById('al-confirm-msg').textContent   = msg;
        document.getElementById('al-confirm-ok').textContent    = okLabel || 'Clear all';
        _alConfirmCallback = callback;
        var skipRow = _alEnsureSkipCheckboxRow();
        if (skipRow) {
            skipRow.style.display = showSkipOption ? 'flex' : 'none';
            var box = document.getElementById('al-confirm-skip-checkbox');
            if (box) box.checked = false;
        }
        document.getElementById('al-confirm-overlay').classList.add('open');
    };
    window.alConfirmClose = function() {
        document.getElementById('al-confirm-overlay').classList.remove('open');
        _alConfirmCallback = null;
    };
    window.alConfirmClear = function() {
        var skipBox = document.getElementById('al-confirm-skip-checkbox');
        if (skipBox && skipBox.checked) _alSkipDeleteConfirm = true;
        if (_alConfirmCallback) _alConfirmCallback();
        alConfirmClose();
        _alResyncKbAfterDelete();
    };
    window.alDeleteConfirm = function(idx, ticker) {
        if (_alSkipDeleteConfirm) {
            alDelete(idx);
            _alResyncKbAfterDelete();
            return;
        }
        alConfirmOpen(
            'Remove alert?',
            'The price alert for ' + ticker + ' will be permanently removed.',
            function() { alDelete(idx); },
            'Delete',
            true
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
        if (document.getElementById('al-confirm-overlay').classList.contains('open')) return; // handled by the confirm-modal listeners above
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
            // The alerts chart owns Delete while a line is selected (or the draw tool has lines to remove).
            // Without this, the highlighted alert row ALSO got the key and opened "Remove alert?" for a plain
            // drawn line. This listener is registered before the chart's own handler, so the chart's selection
            // state is still intact when we check it here.
            if (_alSelectedTrendlineIdx !== -1 || _alSelectedVwapIdx !== -1 ||
                (_alTrendlineMode && _alTrendlines.length)) return;
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
        btn.style.background  = alMcActive ? 'var(--bg-accent-active-2)' : '';
        btn.style.borderColor = alMcActive ? 'var(--accent)' : '';
        btn.style.color       = alMcActive ? 'var(--accent-strong)' : '';
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
