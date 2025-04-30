import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:frontend/core/network/websocket_service.dart'; // Adjust import path if needed
import 'package:frontend/features/stock_ticker/domain/stock_data.dart'; // Adjust import path if needed

enum ConnectionStatus { disconnected, connecting, connected, error }

class StockTickerProvider with ChangeNotifier {
  final WebSocketService _webSocketService;
  StreamSubscription? _messageSubscription;

  ConnectionStatus _connectionStatus = ConnectionStatus.disconnected;
  String? _errorMessage;
  final Set<String> _subscribedTokens = {};
  final Map<String, StockData> _stockDataMap = {};

  // --- Getters for UI ---
  ConnectionStatus get connectionStatus => _connectionStatus;
  String? get errorMessage => _errorMessage;
  Set<String> get subscribedTokens => Set.unmodifiable(_subscribedTokens); // Read-only view
  Map<String, StockData> get stockDataMap => Map.unmodifiable(_stockDataMap); // Read-only view

  StockTickerProvider(this._webSocketService) {
    _listenToMessages();
  }

  // --- Internal Logic ---

  void _listenToMessages() {
    _messageSubscription = _webSocketService.messages.listen((message) {
      final type = message['type'] as String?;
      switch (type) {
        case 'status':
          _handleStatusUpdate(message);
          break;
        case 'subscription_ack':
          _handleSubscriptionAck(message);
          break;
        case 'unsubscription_ack':
          _handleUnsubscriptionAck(message);
          break;
        case 'stock_update':
          _handleStockUpdate(message);
          break;
        case 'error':
           _handleErrorUpdate(message);
          break;
        default:
          debugPrint('Received unknown message type: $type');
      }
    }, onError: (error) {
      debugPrint('Error in WebSocket message stream: $error');
      _updateStatus(ConnectionStatus.error, error.toString());
    });
  }

  void _handleStatusUpdate(Map<String, dynamic> message) {
     final status = message['status'] as String?;
     final msg = message['message'] as String?;
     switch (status) {
       case 'connected':
         _updateStatus(ConnectionStatus.connected);
         // Optional: Resubscribe to tokens if connection was lost and re-established
         _resubscribeTokens();
         break;
       case 'disconnected':
          _updateStatus(ConnectionStatus.disconnected);
          break;
       case 'error':
          _updateStatus(ConnectionStatus.error, msg ?? 'Unknown connection error');
          break;
       default:
          debugPrint('Unknown status received: $status');
     }
  }

   void _handleSubscriptionAck(Map<String, dynamic> message) {
     final stock = message['stock'] as String?;
     if (stock != null) {
       _subscribedTokens.add(stock);
       debugPrint('Subscription acknowledged for: $stock');
       notifyListeners(); // Notify UI about subscription change
     }
   }

   void _handleUnsubscriptionAck(Map<String, dynamic> message) {
     final stock = message['stock'] as String?;
     if (stock != null) {
       _subscribedTokens.remove(stock);
       _stockDataMap.remove(stock); // Remove data when unsubscribed
       debugPrint('Unsubscription acknowledged for: $stock');
       notifyListeners(); // Notify UI about subscription and data change
     }
   }

   void _handleStockUpdate(Map<String, dynamic> message) {
     final data = message['data'] as Map<String, dynamic>?;
     if (data != null) {
       try {
         final stockData = StockData.fromJson(data);
         // Only update if the token is actually subscribed (defensive check)
         if (_subscribedTokens.contains(stockData.stockName)) {
            _stockDataMap[stockData.stockName] = stockData;
            notifyListeners(); // Notify UI about data update
         } else {
            debugPrint('Received update for unsubscribed stock: ${stockData.stockName}');
         }
       } catch (e) {
         debugPrint('Failed to parse stock_update data: $data. Error: $e');
          _updateStatus(ConnectionStatus.error, 'Failed to parse stock data');
       }
     }
   }

   void _handleErrorUpdate(Map<String, dynamic> message) {
      final msg = message['message'] as String?;
      debugPrint('Received error message from server: $msg');
      // Decide how to handle server-side errors (e.g., display in UI)
      _updateStatus(ConnectionStatus.error, msg ?? 'Unknown server error');
   }


  void _updateStatus(ConnectionStatus status, [String? message]) {
    _connectionStatus = status;
    _errorMessage = message;
    if (status != ConnectionStatus.connected) {
      // Consider clearing data or subscriptions on persistent errors/disconnects
      // _subscribedTokens.clear();
      // _stockDataMap.clear();
    }
    notifyListeners();
  }

  void _resubscribeTokens() {
     if (_connectionStatus == ConnectionStatus.connected) {
        final tokensToResubscribe = Set<String>.from(_subscribedTokens); // Copy set
        debugPrint('Resubscribing to tokens: $tokensToResubscribe');
        for (final token in tokensToResubscribe) {
           _webSocketService.subscribe(token);
        }
     }
  }

  // --- Public Methods for UI Actions ---

  Future<void> connect() async {
    if (_connectionStatus == ConnectionStatus.connecting || _connectionStatus == ConnectionStatus.connected) {
      return;
    }
    _updateStatus(ConnectionStatus.connecting);
    await _webSocketService.connect();
    // Status will be updated via the message listener upon success/failure
  }

  void disconnect() {
    _webSocketService.disconnect();
    _updateStatus(ConnectionStatus.disconnected);
    // Clear state on manual disconnect
    _subscribedTokens.clear();
    _stockDataMap.clear();
    notifyListeners();
  }

  void subscribe(String stockToken) {
    if (stockToken.isNotEmpty && _connectionStatus == ConnectionStatus.connected) {
      _webSocketService.subscribe(stockToken);
      // Add optimistically? Or wait for ack? Waiting for ack is safer.
      // _subscribedTokens.add(stockToken);
      // notifyListeners();
    } else if (_connectionStatus != ConnectionStatus.connected) {
       _updateStatus(ConnectionStatus.error, 'Cannot subscribe: Not connected');
    }
  }

  void unsubscribe(String stockToken) {
    if (stockToken.isNotEmpty && _connectionStatus == ConnectionStatus.connected) {
      _webSocketService.unsubscribe(stockToken);
      // Remove optimistically? Or wait for ack? Waiting for ack is safer.
      // _subscribedTokens.remove(stockToken);
      // _stockDataMap.remove(stockToken);
      // notifyListeners();
    } else if (_connectionStatus != ConnectionStatus.connected) {
       _updateStatus(ConnectionStatus.error, 'Cannot unsubscribe: Not connected');
    }
  }

  // --- Cleanup ---

  @override
  void dispose() {
    debugPrint('Disposing StockTickerProvider...');
    _messageSubscription?.cancel();
    _webSocketService.dispose(); // Dispose the service if owned by this provider
    super.dispose();
    debugPrint('StockTickerProvider disposed.');
  }
}
