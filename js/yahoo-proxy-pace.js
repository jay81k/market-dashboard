// ── Shared Yahoo-proxy launch pacing ─────────────────────────────────────
// multichart.js, market.js, and state.js all send requests through
// yahoo-proxy.jay69k.workers.dev, each with its own queue/retry logic suited
// to what it fetches — but each used to track launch timing and 429 cooldowns
// independently. That meant one script's well-paced requests could still get
// 429'd by another script's burst landing on the same upstream Yahoo host at
// the same moment, since neither knew the other existed. This file is the one
// thing all three now share: a single clock and a single cooldown, so a burst
// anywhere backs off everywhere instead of three pacers that can't see each
// other.
//
// This file must load BEFORE multichart.js, market.js, and state.js.
//
// Each consumer keeps its own launch-spacing preference (how often IT is
// willing to fire) and its own queue/concurrency/retry logic — only the
// clock (lastLaunchAt) and the 429 cooldown are shared, since only one of
// each can coherently exist across every caller.
(function() {
    var pace = { lastLaunchAt: 0, recent429s: [], cooldownUntil: 0 };

    var WINDOW_429    = 3000; // ms window for counting recent 429s
    var THRESHOLD_429 = 3;    // this many 429s inside the window trips the cooldown
    var COOLDOWN_MS   = 6000; // pause all new launches this long once tripped

    function register429() {
        var now = Date.now();
        pace.recent429s.push(now);
        while (pace.recent429s.length && now - pace.recent429s[0] > WINDOW_429) {
            pace.recent429s.shift();
        }
        if (pace.recent429s.length >= THRESHOLD_429) {
            pace.cooldownUntil = now + COOLDOWN_MS;
            pace.recent429s = [];
        }
    }

    window.yahooProxyPace = {
        cooldownUntil: function() { return pace.cooldownUntil; },
        lastLaunchAt:  function() { return pace.lastLaunchAt; },
        markLaunched:  function() { pace.lastLaunchAt = Date.now(); },
        register429:   register429,
    };
})();
