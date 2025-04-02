/*
Documentation:
---------------
Class: ApiErrorHandler
Description:
  A helper class that extracts human-readable messages from DioExceptions and constructs AppExceptions for invalid responses.

Methods:
  • handleDioError(e):
      - Extracts an error message from a DioException.
      - Example: ApiErrorHandler.handleDioError(error) returns a string message.
      
  • _getDefaultMessage(e):
      - Returns a default error message based on the type of DioException.
      
  • handleInvalidResponse(response):
      - Constructs an AppException when a server response is invalid.
      - Example: Throws an exception if response.statusCode is not as expected.
*/

// Code:
import 'package:dio/dio.dart';
import 'package:frontend/core/utils/exceptions/app_exceptions.dart';

/// A helper class that extracts human-readable messages from DioExceptions
/// and constructs AppExceptions for invalid responses.
class ApiErrorHandler {
  /// Extracts an error message from a DioException.
  static String handleDioError(DioException e) {
    final responseData = e.response?.data;
    return (responseData is Map<String, dynamic>
            ? responseData['detail'] as String?
            : null) ??
        _getDefaultMessage(e);
  }

  /// Returns a default error message based on the error type.
  static String _getDefaultMessage(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
        return 'Connection timeout';
      case DioExceptionType.sendTimeout:
        return 'Send timeout';
      case DioExceptionType.receiveTimeout:
        return 'Receive timeout';
      case DioExceptionType.badCertificate:
        return 'Bad Certificate';
      case DioExceptionType.badResponse:
        return e.response != null
            ? 'Invalid server response (Status code: ${e.response!.statusCode})'
            : 'Invalid server response';
      case DioExceptionType.cancel:
        return 'Request cancelled';
      case DioExceptionType.connectionError:
        return 'Connection error';
      default:
        return e.message ?? 'Unknown network error';
    }
  }

  /// Constructs an AppException from an invalid response.
  static AppException handleInvalidResponse(Response response) {
    final method = response.requestOptions.method;
    final path = response.requestOptions.path;
    return AppException(
        'Invalid response (${response.statusCode}) for $method request to $path');
  }
}
