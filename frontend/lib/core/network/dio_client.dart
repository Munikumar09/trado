/*
Documentation:
---------------
Class: DioClient
Description:
  Provides a pre-configured Dio instance for HTTP requests.
  Sets base URL, timeouts, and registers interceptors for authentication and logging.

Methods:
  • DioClient(secureStorage, ref, {connectionTimeout, receiveTimeout}):
      - Initializes Dio with options and calls _configureInterceptors().
      - Example: final client = DioClient(secureStorage, ref);
      
  • dio:
      - Getter that returns the configured Dio instance.
      
  • _configureInterceptors():
      - Adds AuthInterceptor and, in debug mode, PrettyDioLogger.
      - Example: print(client.dio.interceptors);
*/

// Code:
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/constants/api_endpoints.dart';
import 'package:frontend/core/network/interceptor/auth_interceptor.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

/// DioClient provides a pre-configured Dio HTTP client.
class DioClient {
  final int connectionTimeout;
  final int receiveTimeout;
  final Dio _dio;
  final SecureStorageService _secureStorage;
  final Ref ref;

  /// Constructs DioClient with secure storage and timeout configurations.
  DioClient(
    this._secureStorage,
    this.ref, {
    this.connectionTimeout = 5000,
    this.receiveTimeout = 5000,
  }) : _dio = Dio(
          BaseOptions(
            baseUrl: ApiEndpoints.baseUrl,
            connectTimeout: Duration(milliseconds: connectionTimeout),
            receiveTimeout: Duration(milliseconds: receiveTimeout),
            contentType: 'application/json',
          ),
        ) {
    _configureInterceptors();
  }

  /// Returns the configured Dio instance.
  Dio get dio => _dio;

  /// Configures interceptors: AuthInterceptor for auth management and PrettyDioLogger in debug mode.
  void _configureInterceptors() {
    final interceptors = <Interceptor>[
      AuthInterceptor(_secureStorage, _dio, ref), // Manages auth tokens.
    ];

    if (kDebugMode) {
      interceptors.add(
        PrettyDioLogger(
          requestHeader: true,
          requestBody: true,
          responseHeader: true,
          error: true,
          maxWidth: 80,
        ), // Logs requests/responses in debug mode.
      );
    }

    _dio.interceptors.addAll(interceptors);
  }
}
