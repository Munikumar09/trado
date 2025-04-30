import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart'; // For kDebugMode
import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

// TODO: Make this configurable (e.g., via environment variables or settings)
const String _defaultWebSocketUrl = 'ws://localhost:8080';

class WebSocketService {
  WebSocketChannel? _channel;
  final StreamController<Map<String, dynamic>> _messageStreamController =
      StreamController.broadcast();
  StreamSubscription? _channelSubscription;
  bool _isConnected = false;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  final int _maxReconnectAttempts = 5;
  final Duration _reconnectDelay = const Duration(seconds: 5);

  // Public stream for widgets/providers to listen to
  Stream<Map<String, dynamic>> get messages => _messageStreamController.stream;
  bool get isConnected => _isConnected;

  // --- Connection Management ---

  Future<void> connect({String url = _defaultWebSocketUrl}) async {
    if (_isConnected && _channel != null) {
      debugPrint('WebSocket already connected.');
      return;
    }
    debugPrint('Attempting to connect to WebSocket: $url');
    _cancelReconnectTimer(); // Cancel any pending reconnection attempts

    try {
      // Use IOWebSocketChannel for mobile/desktop, consider conditional import for web
      _channel = IOWebSocketChannel.connect(Uri.parse(url));
      _isConnected = true;
      _reconnectAttempts = 0; // Reset attempts on successful connection
      debugPrint('WebSocket connected successfully.');

      _channelSubscription = _channel!.stream.listen(
        _onMessageReceived,
        onDone: _handleDisconnect,
        onError: _handleError,
        cancelOnError: false, // Keep listening even after errors
      );

      // Notify listeners about connection status change (optional)
      _messageStreamController.add({'type': 'status', 'status': 'connected'});

    } catch (e) {
      debugPrint('WebSocket connection error: $e');
      _isConnected = false;
      _scheduleReconnect(url: url); // Schedule reconnection on initial failure
      // Notify listeners about connection status change (optional)
      _messageStreamController.add({'type': 'status', 'status': 'error', 'message': e.toString()});
    }
  }

  void disconnect() {
    debugPrint('Disconnecting WebSocket...');
    _cancelReconnectTimer();
    _channelSubscription?.cancel();
    _channel?.sink.close();
    _channel = null;
    _isConnected = false;
    debugPrint('WebSocket disconnected.');
     // Notify listeners about connection status change (optional)
    _messageStreamController.add({'type': 'status', 'status': 'disconnected'});
  }

  void _handleDisconnect() {
    debugPrint('WebSocket connection closed by server.');
    if (_isConnected) { // Only attempt reconnect if we thought we were connected
      _isConnected = false;
      // Notify listeners about connection status change (optional)
      _messageStreamController.add({'type': 'status', 'status': 'disconnected'});
      _scheduleReconnect(); // Attempt to reconnect
    }
  }

  void _handleError(error) {
    debugPrint('WebSocket error: $error');
    // Errors might or might not cause disconnection. onDone handles actual disconnects.
    // We might still be connected after some errors.
    // Notify listeners about the error
     _messageStreamController.add({'type': 'status', 'status': 'error', 'message': error.toString()});
    // Consider if specific errors should trigger disconnect/reconnect logic
    // For now, rely on onDone for disconnects.
  }

  // --- Reconnection Logic ---

  void _scheduleReconnect({String url = _defaultWebSocketUrl}) {
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      debugPrint('Max reconnection attempts reached. Giving up.');
      return;
    }
    _reconnectAttempts++;
    final delay = Duration(seconds: _reconnectDelay.inSeconds * _reconnectAttempts); // Exponential backoff (simple)
    debugPrint('Scheduling WebSocket reconnection attempt $_reconnectAttempts/$_maxReconnectAttempts in ${delay.inSeconds} seconds...');

    _cancelReconnectTimer();
    _reconnectTimer = Timer(delay, () => connect(url: url));
  }

  void _cancelReconnectTimer() {
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
  }


  // --- Message Handling ---

  void _onMessageReceived(dynamic message) {
    // Assuming messages are JSON strings
    if (message is String) {
      try {
        final decodedMessage = jsonDecode(message) as Map<String, dynamic>;
        if (kDebugMode) {
          print('WebSocket message received: $decodedMessage');
        }
        // Add the decoded message to the stream for listeners
        _messageStreamController.add(decodedMessage);
      } catch (e) {
        debugPrint('Failed to decode JSON message: $message. Error: $e');
        // Optionally add an error message to the stream
         _messageStreamController.add({'type': 'error', 'message': 'Failed to decode message', 'original': message});
      }
    } else {
      debugPrint('Received non-string message: $message');
       _messageStreamController.add({'type': 'error', 'message': 'Received non-string message', 'original': message.toString()});
    }
  }

  void _sendMessage(Map<String, dynamic> message) {
    if (_channel != null && _isConnected) {
      try {
        final encodedMessage = jsonEncode(message);
        if (kDebugMode) {
          print('Sending WebSocket message: $encodedMessage');
        }
        _channel!.sink.add(encodedMessage);
      } catch (e) {
         debugPrint('Failed to encode or send message: $message. Error: $e');
         // Notify listeners about the error
         _messageStreamController.add({'type': 'error', 'message': 'Failed to send message', 'details': e.toString()});
      }
    } else {
      debugPrint('Cannot send message: WebSocket not connected.');
       _messageStreamController.add({'type': 'error', 'message': 'Cannot send message: Not connected'});
    }
  }

  // --- Public Methods for Actions ---

  void subscribe(String stockToken) {
    if (stockToken.isEmpty) return;
    _sendMessage({'action': 'subscribe', 'stock': stockToken.toUpperCase()});
  }

  void unsubscribe(String stockToken) {
     if (stockToken.isEmpty) return;
    _sendMessage({'action': 'unsubscribe', 'stock': stockToken.toUpperCase()});
  }

  // --- Cleanup ---

  void dispose() {
    debugPrint('Disposing WebSocketService...');
    disconnect(); // Ensure everything is closed
    _messageStreamController.close();
    debugPrint('WebSocketService disposed.');
  }
}
