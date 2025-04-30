import 'package:flutter/foundation.dart';

@immutable // Good practice for model classes used in state management
class StockData {
  final String stockName;
  final double stockPrice;
  final String lastTradedTimestamp; // Keep original string for display if needed
  final DateTime processedTimestampUtc;
  final DateTime cacheUpdatedAtUtc;

  const StockData({
    required this.stockName,
    required this.stockPrice,
    required this.lastTradedTimestamp,
    required this.processedTimestampUtc,
    required this.cacheUpdatedAtUtc,
  });

  // Factory constructor to create StockData from the JSON map received
  factory StockData.fromJson(Map<String, dynamic> json) {
    // Basic validation and type checking
    final stockName = json['stock_name'] as String? ?? 'UNKNOWN';
    final stockPrice = (json['stock_price'] as num?)?.toDouble() ?? 0.0;
    final lastTradedTimestamp = json['last_traded_timestamp'] as String? ?? '';
    
    DateTime processedTimestamp;
    try {
      processedTimestamp = DateTime.parse(json['processed_timestamp_utc'] as String? ?? '');
    } catch (_) {
      processedTimestamp = DateTime.fromMillisecondsSinceEpoch(0, isUtc: true); // Default fallback
    }

     DateTime cacheUpdatedAt;
    try {
      cacheUpdatedAt = DateTime.parse(json['cache_updated_at_utc'] as String? ?? '');
    } catch (_) {
      cacheUpdatedAt = DateTime.fromMillisecondsSinceEpoch(0, isUtc: true); // Default fallback
    }

    return StockData(
      stockName: stockName,
      stockPrice: stockPrice,
      lastTradedTimestamp: lastTradedTimestamp,
      processedTimestampUtc: processedTimestamp,
      cacheUpdatedAtUtc: cacheUpdatedAt,
    );
  }

  // Optional: Add toString, equality, hashCode for debugging/testing
  @override
  String toString() {
    return 'StockData(stockName: $stockName, stockPrice: $stockPrice, processedTimestampUtc: $processedTimestampUtc)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
  
    return other is StockData &&
      other.stockName == stockName &&
      other.stockPrice == stockPrice &&
      other.lastTradedTimestamp == lastTradedTimestamp &&
      other.processedTimestampUtc == processedTimestampUtc &&
      other.cacheUpdatedAtUtc == cacheUpdatedAtUtc;
  }

  @override
  int get hashCode {
    return stockName.hashCode ^
      stockPrice.hashCode ^
      lastTradedTimestamp.hashCode ^
      processedTimestampUtc.hashCode ^
      cacheUpdatedAtUtc.hashCode;
  }
}
