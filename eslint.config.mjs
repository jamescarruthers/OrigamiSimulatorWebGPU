import globals from 'globals';

// ESLint is used here primarily as a STATIC verification backstop for the
// WebGL->WebGPU migration's ES-module conversion (Phase 0). With sourceType
// 'module' + `no-undef`, it flags any cross-module symbol that was forgotten
// in an import (which Rollup would otherwise silently leave as a runtime
// ReferenceError) — letting us validate the module graph without a browser.
export default [
  {
    ignores: ['dependencies/**', 'dist/**', 'node_modules/**', 'tests/**', '*.config.mjs'],
  },
  {
    files: ['js/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        // App-wide mutable state object; assigned on window in js/main.js and
        // read as a free global throughout.
        globals: 'writable',
        escape: 'readonly',
        unescape: 'readonly',
        // Third-party libraries loaded as classic <script> tags (they install
        // these on window before the module entry runs).
        THREE: 'readonly',
        $: 'readonly',
        jQuery: 'readonly',
        _: 'readonly',
        numeric: 'readonly',
        earcut: 'readonly',
        cdt2d: 'readonly',
        svgpath: 'readonly',
        fold: 'readonly',
        // Global browserify-style require shim installed by dependencies/fold.js
        // (resolves 'fold', 'svgpath', 'cdt2d').
        require: 'readonly',
        saveAs: 'readonly',
        CCapture: 'readonly',
        GIF: 'readonly',
        dat: 'readonly',
        WEBVR: 'readonly',
        // Global from dependencies/binary_stl_writer.js (classic script).
        geometryToSTLBin: 'readonly',
        // Google Analytics shim defined in the inline <script> in index.html.
        gtag: 'readonly',
      },
    },
    rules: {
      'no-undef': 'error',
      'no-unused-vars': 'off',
    },
  },
];
