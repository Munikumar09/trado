/*
Documentation:
---------------
Class: ApiCallHandler
Description:
  Centralizes execution of API calls with comprehensive error handling and logging.
  Uses Dio for HTTP requests, validates responses, and maps exceptions to custom AppExceptions.

Methods:
  • handleApiCall<T>({call, exception, operationName, suppressError}):
      - Executes the provided API call and handles errors.
      - Example:
          final result = await apiCallHandler.handleApiCall(
            call: () => dio.get('/users'),
            exception: (msg) => CustomException(msg),
            operationName: 'Fetch Users',
          );
      
  • validateResponse(response, {successStatus}):
      - Validates the response status code.
      - Example: final validResponse = apiCallHandler.validateResponse(response);
*/

// Code:
import 'package:dio/dio.dart';
import 'package:frontend/core/utils/exceptions/app_exceptions.dart';
import 'package:frontend/core/utils/handlers/api_error_handler.dart';
import 'package:logger/logger.dart';

/// ApiCallHandler centralizes API call execution with error handling.
class ApiCallHandler {
  final Logger _logger;

  /// Constructs an ApiCallHandler with an optional logger.
  ApiCallHandler({Logger? logger})
      : _logger = logger ??
            Logger(
              printer: PrettyPrinter(
                methodCount: 0,
                errorMethodCount: 5,
                lineLength: 120,
                colors: true,
                printEmojis: true,
                printTime: false,
              ),
            );

  /// Executes an API call, logging outcome and handling exceptions.
  Future<T> handleApiCall<T>({
    required Future<T> Function() call,
    required AppException Function(String) exception,
    required String operationName,
    bool suppressError = false, // Set true to return null instead of throwing.
  }) async {
    try {
      final result = await call();
      _logger.i('$operationName succeeded');
      return result;
    } on DioException catch (e, stackTrace) {
      final errorMessage = ApiErrorHandler.handleDioError(e);
      _logger.e('$operationName failed: $errorMessage', e, stackTrace);
      if (!suppressError) {
        throw exception(errorMessage);
      } else {
        return Future<T>.value(null);
      }
    } on AppException catch (e, stackTrace) {
      _logger.e('$operationName failed: ${e.message}', e, stackTrace);
      if (!suppressError) {
        rethrow;
      } else {
        return Future<T>.value(null);
      }
    } catch (e, stackTrace) {
      _logger.e('Unexpected error during $operationName: $e', e, stackTrace);
      if (!suppressError) {
        throw exception('An unexpected error occurred');
      } else {
        return Future<T>.value(null);
      }
    }
  }

  /// Validates the HTTP response.
  /// Throws an exception if the status code does not equal [successStatus].
  Response validateResponse(Response response, {int successStatus = 200}) {
    if (response.statusCode != successStatus) {
      throw ApiErrorHandler.handleInvalidResponse(response);
    }
    return response;
  }
}
