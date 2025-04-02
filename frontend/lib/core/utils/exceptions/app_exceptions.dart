/*
Documentation:
---------------
Module: Application Exceptions
Description:
  Provides a base exception class for the application along with other common exceptions 
  such as cache and input validation failures.
  
Classes:
  • AppException: Base exception for all application-specific errors.
  • CacheException: Thrown when caching fails.
  • InputValidationException: Thrown when user input validation fails.
*/

/// Base exception for all application-specific errors.
class AppException implements Exception {
  final String message;
  final StackTrace? stackTrace;

  /// Constructs an [AppException] with the provided message and an optional stack trace.
  AppException(this.message, [this.stackTrace]);

  @override
  String toString() => 'AppException: $message';
}

// Other Exceptions (You can add more as needed)

/// Thrown when a caching error occurs.
class CacheException extends AppException {
  /// Constructs a [CacheException] with the provided message and optional stack trace.
  CacheException(super.message, [super.stackTrace]);
}

/// Thrown when user input validation fails.
class InputValidationException extends AppException {
  /// Constructs an [InputValidationException] with the provided message and optional stack trace.
  InputValidationException(super.message, [super.stackTrace]);
}
