/*
Documentation:
---------------
Class: AuthInterceptor
Description:
  Intercepts HTTP requests to attach an Authorization header if available.
  For 401 errors, triggers token refresh, queues further requests during refresh,
  and retries or rejects those requests accordingly.

Methods:
  • onRequest(options, handler):
      - Attaches the Authorization header with a Bearer token.
      - Example: Requests get the header if an access token exists.
      
  • onError(err, handler):
      - Handles 401 errors and initiates token refresh.
      - Example: Retries queued requests on successful refresh.
      
  • _attemptTokenRefresh(err):
      - Attempts to refresh the access token.
      
  • _refreshAccessToken(refreshToken):
      - Sends a POST request to refresh the token.
      
  • _queueRequest(options, handler):
      - Queues requests during token refresh.
      
  • _retryQueuedRequests(newAccessToken):
      - Retries all queued requests with the new token.
      
  • _rejectQueuedRequests(originalError):
      - Rejects all queued requests if refresh fails.
      
  • _retryRequest(options, newAccessToken, handler):
      - Retries a single request with updated Authorization header.
      
  • _clearAuthDataAndLogout(error):
      - Clears stored tokens and triggers logout.
*/

// Code:
import 'dart:async';
import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/constants/api_endpoints.dart';
import 'package:frontend/core/constants/storage_keys.dart';
import 'package:frontend/features/auth/application/providers/auth_providers.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:logger/logger.dart';

/// AuthInterceptor handles attaching the Bearer token and token refresh logic.
class AuthInterceptor extends Interceptor {
  static const _authorizationHeader = 'Authorization';
  static const _bearerPrefix = 'Bearer ';
  static const refreshTokenEndpoint = ApiEndpoints.refreshToken;

  final SecureStorageService _secureStorage;
  final Dio _dio;
  final Ref ref;
  final Logger _logger = Logger();

  bool _isRefreshing = false;
  static const _maxQueueSize = 50;
  final List<_QueuedRequest> _requestQueue = [];

  AuthInterceptor(this._secureStorage, this._dio, this.ref);

  /// onRequest attaches the Authorization header if an access token exists.
  @override
  Future<void> onRequest(
      RequestOptions options, RequestInterceptorHandler handler) async {
    final tokens = await _secureStorage.getTokens();
    if (tokens.accessToken != null) {
      options.headers[_authorizationHeader] =
          _bearerPrefix + tokens.accessToken!;
      _logger.i("onRequest: Adding Authorization header");
    }
    _logger.d("onRequest: URL: ${options.uri}, Headers: ${options.headers}");
    return handler.next(options);
  }

  /// onError processes 401 errors by attempting to refresh the token.
  @override
  Future<void> onError(
      DioException err, ErrorInterceptorHandler handler) async {
    final response = err.response;
    _logger.e(
        "onError triggered. URL: ${err.requestOptions.uri}, Status Code: ${response?.statusCode}, _isRefreshing: $_isRefreshing, Error: ${err.message}");
    if (response?.statusCode != 401) return handler.next(err);
    if (err.requestOptions.path.contains(refreshTokenEndpoint)) {
      _logger.e(
          "onError: Refresh token request failed. Logging out and rejecting request.");
      await _clearAuthDataAndLogout(err);
      return handler.reject(err);
    }
    if (_isRefreshing) {
      _logger.i("onError: Refresh already in progress. Queuing request.");
      _queueRequest(err.requestOptions, handler);
      return;
    }
    _isRefreshing = true;
    _logger
        .i("onError: Attempting token refresh. Setting _isRefreshing to true.");
    try {
      final newAccessToken = await _attemptTokenRefresh(err);
      await _retryQueuedRequests(newAccessToken!);
      _retryRequest(err.requestOptions, newAccessToken, handler);
    } catch (e) {
      _logger.e("onError: Error during refresh attempt: $e", e);
      await _rejectQueuedRequests(err);
      handler.reject(err);
    } finally {
      _isRefreshing = false;
      _logger.i(
          "onError: Refresh attempt complete. Resetting _isRefreshing to false.");
    }
  }

  /// Attempts to refresh the access token.
  Future<String?> _attemptTokenRefresh(DioException err) async {
    _logger.i("_attemptTokenRefresh: Called");
    final tokens = await _secureStorage.getTokens();
    if (tokens.refreshToken == null) {
      _logger.e('_attemptTokenRefresh: No refresh token. Clearing data.');
      await _clearAuthDataAndLogout(err);
      throw DioException(
        requestOptions: err.requestOptions,
        response: err.response,
        type: DioExceptionType.unknown,
        error: "No refresh token available",
      );
    }
    _logger.i("_attemptTokenRefresh: Retrieved refresh token");
    final newAccessToken = await _refreshAccessToken(tokens.refreshToken!);
    if (newAccessToken == null) {
      _logger.e(
          '_attemptTokenRefresh: New Access Token is null. Throwing exception.');
      await _clearAuthDataAndLogout(err);
      throw DioException(
          requestOptions: err.requestOptions,
          response: err.response,
          type: err.type,
          error: "New Access Token is null");
    }
    _logger.i("_attemptTokenRefresh: Successfully retrieved new access token.");
    return newAccessToken;
  }

  /// Sends a POST request to refresh the access token.
  Future<String?> _refreshAccessToken(String refreshToken) async {
    _logger.i("_refreshAccessToken: Attempting to refresh token.");
    try {
      final String errorMessage;
      final response = await _dio.post(
        refreshTokenEndpoint,
        queryParameters: {StorageKeys.refreshToken: refreshToken},
      );
      if (response.statusCode != null &&
          response.statusCode! >= 200 &&
          response.statusCode! < 300) {
        final newAccessToken =
            response.data?[StorageKeys.accessToken] as String?;
        if (newAccessToken != null) {
          _logger.i(
              "_refreshAccessToken: Successfully retrieved new access token from server.");
          await _secureStorage.saveTokens(
              accessToken: newAccessToken, refreshToken: refreshToken);
          return newAccessToken;
        }
        errorMessage =
            response.data?['detail'] as String? ?? 'Access Token is null';
      } else {
        errorMessage =
            response.data?['detail'] as String? ?? 'Refresh token failed';
      }
      throw DioException(
          requestOptions: RequestOptions(),
          response: response,
          type: DioExceptionType.badResponse,
          error: errorMessage);
    } on DioException catch (e) {
      _logger.e("_refreshAccessToken: DioException: ${e.message}", e);
      await _clearAuthDataAndLogout(e);
      rethrow;
    }
  }

  /// Queues a request during token refresh.
  void _queueRequest(RequestOptions options, ErrorInterceptorHandler handler) {
    if (_requestQueue.length >= _maxQueueSize) {
      _logger.w("Queue size limit reached. Rejecting request: ${options.uri}");
      handler.reject(DioException(
        requestOptions: options,
        error: "Queue size limit reached",
        type: DioExceptionType.unknown,
      ));
      return;
    }
    _logger.i("Queuing request: ${options.uri}");
    _requestQueue.add(_QueuedRequest(options, handler));
  }

  /// Retries all queued requests with the new access token.
  Future<void> _retryQueuedRequests(String newAccessToken) async {
    _logger
        .i("Retrying queued requests. Queue length: ${_requestQueue.length}");
    for (final queued in _requestQueue) {
      _logger.i("Retrying request: ${queued.requestOptions.uri}");
      await _retryRequest(
          queued.requestOptions, newAccessToken, queued.handler);
    }
    _requestQueue.clear();
  }

  /// Rejects all queued requests using the original error.
  Future<void> _rejectQueuedRequests(DioException originalError) async {
    _logger.e(
        "Rejecting all queued requests. Queue length: ${_requestQueue.length}");
    for (final queued in _requestQueue) {
      _logger.e("Rejecting request: ${queued.requestOptions.uri}");
      queued.handler.reject(originalError);
    }
    _requestQueue.clear();
  }

  /// Retries a single request with an updated Authorization header.
  Future<void> _retryRequest(RequestOptions options, String newAccessToken,
      ErrorInterceptorHandler handler) async {
    options.headers[_authorizationHeader] = _bearerPrefix + newAccessToken;
    try {
      final response = await _dio.fetch(options);
      handler.resolve(response);
    } catch (e) {
      handler.reject(e as DioException);
    }
  }

  /// Clears stored auth data and triggers logout.
  Future<void> _clearAuthDataAndLogout(DioException error) async {
    ref.read(authNotifierProvider.notifier).logout();
    _isRefreshing = false;
    _logger.e(
        "_clearAuthDataAndLogout: Clearing auth data and logging out. Error: ${error.response?.data ?? error.message}");
  }
}

class _QueuedRequest {
  final RequestOptions requestOptions;
  final ErrorInterceptorHandler handler;
  _QueuedRequest(this.requestOptions, this.handler);
}
