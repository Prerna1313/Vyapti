/**
 * VYAPTI Backend Bridge
 * ====================
 * Connects the RF dashboard frontend pages to the Vyapti FastAPI backend
 * (http://localhost:8000) via REST and WebSocket.
 *
 * Usage
 * -----
 *   Load this script BEFORE app.js / app-receiver.js.
 *   The global `window.VyaptiBridge` is available immediately.
 *
 *   window.VyaptiBridge.start()   → POST /api/start
 *   window.VyaptiBridge.pause()   → POST /api/pause
 *   window.VyaptiBridge.stop()    → POST /api/stop
 *   window.VyaptiBridge.reset()   → POST /api/reset
 *   window.VyaptiBridge.status()  → GET  /api/status  (returns Promise<payload>)
 *
 *   WebSocket frames are re-broadcast as a CustomEvent on document:
 *     document.addEventListener('vyapti-telemetry', e => {
 *       const { status, sim_time_s, tick, running, emitter_count } = e.detail;
 *     });
 *
 * The bridge never modifies any existing animation or canvas logic.
 * It is purely a data channel.
 */
(function () {
  'use strict';

  var BASE_URL = 'http://localhost:8000';
  var WS_URL   = 'ws://localhost:8000/ws/telemetry';

  /* ------------------------------------------------------------------ */
  /* Internals                                                            */
  /* ------------------------------------------------------------------ */

  var _socket = null;
  var _reconnectTimer = null;
  var _reconnectDelay = 2000;
  var _connected = false;
  var _lastPayload = null;

  /** Dispatch the telemetry CustomEvent on document so any page can listen. */
  function _dispatch(payload) {
    _lastPayload = payload;
    document.dispatchEvent(new CustomEvent('vyapti-telemetry', { detail: payload }));
  }

  /** Update the small connection badge if present. */
  function _setConnectionBadge(connected) {
    var badge = document.getElementById('vyaptiBridgeBadge');
    if (!badge) return;
    badge.textContent = connected ? '⬤ BACKEND LIVE' : '⬤ BACKEND OFFLINE';
    badge.style.color  = connected ? '#22c55e' : '#ef4444';
  }

  /* ------------------------------------------------------------------ */
  /* WebSocket                                                            */
  /* ------------------------------------------------------------------ */

  function _connectWS() {
    if (_socket && (_socket.readyState === WebSocket.OPEN ||
                    _socket.readyState === WebSocket.CONNECTING)) return;

    try {
      _socket = new WebSocket(WS_URL);
    } catch (e) {
      _scheduleReconnect();
      return;
    }

    _socket.onopen = function () {
      _connected = true;
      _setConnectionBadge(true);
      _reconnectDelay = 2000; // reset back-off
      if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    };

    _socket.onmessage = function (evt) {
      try {
        var data = JSON.parse(evt.data);
        _dispatch(data);
      } catch (_) { /* ignore malformed frames */ }
    };

    _socket.onerror = function () {
      _connected = false;
      _setConnectionBadge(false);
    };

    _socket.onclose = function () {
      _connected = false;
      _setConnectionBadge(false);
      _scheduleReconnect();
    };
  }

  function _scheduleReconnect() {
    if (_reconnectTimer) return;
    _reconnectTimer = setTimeout(function () {
      _reconnectTimer = null;
      _reconnectDelay = Math.min(_reconnectDelay * 1.5, 15000);
      _connectWS();
    }, _reconnectDelay);
  }

  /* ------------------------------------------------------------------ */
  /* REST helpers                                                         */
  /* ------------------------------------------------------------------ */

  function _post(path) {
    return fetch(BASE_URL + path, { method: 'POST' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (payload) {
        _dispatch(payload);
        return payload;
      })
      .catch(function (err) {
        console.warn('[VyaptiBridge] POST ' + path + ' failed:', err.message);
        return null;
      });
  }

  function _get(path) {
    return fetch(BASE_URL + path)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (payload) {
        _dispatch(payload);
        return payload;
      })
      .catch(function (err) {
        console.warn('[VyaptiBridge] GET ' + path + ' failed:', err.message);
        return null;
      });
  }

  /* ------------------------------------------------------------------ */
  /* Public API                                                           */
  /* ------------------------------------------------------------------ */

  window.VyaptiBridge = {
    /** Start the backend simulation. Returns Promise<payload|null>. */
    start:  function () { return _post('/api/start');  },
    /** Pause the backend simulation. Returns Promise<payload|null>. */
    pause:  function () { return _post('/api/pause');  },
    /** Stop the backend simulation. Returns Promise<payload|null>. */
    stop:   function () { return _post('/api/stop');   },
    /** Reset the backend simulation. Returns Promise<payload|null>. */
    reset:  function () { return _post('/api/reset');  },
    /** Fetch current status. Returns Promise<payload|null>. */
    status: function () { return _get('/api/status');  },
    /** True when the WebSocket is connected. */
    get connected() { return _connected; },
    /** Last telemetry payload received (or null). */
    get lastPayload() { return _lastPayload; },
  };

  /* ------------------------------------------------------------------ */
  /* Auto-start WebSocket on page load                                    */
  /* ------------------------------------------------------------------ */

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _connectWS);
  } else {
    _connectWS();
  }

})();
