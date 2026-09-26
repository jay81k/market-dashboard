// ── Light/dark theme toggle ──────────────────────────────────────────────
// Persists the user's choice to localStorage under THEME_STORAGE_KEY.
// The actual dark->light flash-prevention read happens inline in
// index.html's <head>, before this file even loads, so the correct
// theme is already applied by the time this runs.
(function() {
    var STORAGE_KEY = 'dashboard-theme';

    function applyLabel(theme) {
        var el = document.getElementById('theme-toggle-switch');
        if (el) el.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
    }

    // Bridge from CSS custom properties into JS, for canvas and third-party
    // charting libraries that can't read var(--x) themselves. Always reflects
    // whichever theme is currently active — single source of truth stays styles.css.
    window.themeColor = function(varName) {
        return getComputedStyle(document.documentElement).getPropertyValue('--' + varName).trim();
    };

    window.toggleTheme = function() {
        var current = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
        var next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        try { localStorage.setItem(STORAGE_KEY, next); } catch (e) {}
        applyLabel(next);
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
    };

    document.addEventListener('DOMContentLoaded', function() {
        applyLabel(document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark');
    });
})();
