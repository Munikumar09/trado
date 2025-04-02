/*
Documentation:
---------------
Module: Network Exceptions
Description:
  Defines custom exceptions for network-related errors such as server errors,
  unauthorized access, and resource not found. These exceptions extend from AppException.
  
Classes:
  • NetworkException: Base exception for network issues.
  • ServerException: Thrown when a server error occurs.
  • UnauthorizedException: Thrown when access is unauthorized.
  • NotFoundException: Thrown when a resource could not be found.
*/

import 'package:frontend/core/utils/exceptions/app_exceptions.dart';

/// Base exception for network related errors.
class NetworkException extends AppException {
  /// Constructs a [NetworkException] with the provided message and optional stack trace.
  NetworkException(super.message, [super.stackTrace]);
}

/// Thrown when a server error occurs.
class ServerException extends NetworkException {
  /// Constructs a [ServerException] with the provided message and optional stack trace.
  ServerException(super.message, [super.stackTrace]);
}

/// Thrown when access to a resource is unauthorized.
class UnauthorizedException extends NetworkException {
  /// Constructs an [UnauthorizedException] with the provided message and optional stack trace.
  UnauthorizedException(super.message, [super.stackTrace]);
}

/// Thrown when a resource cannot be found.
class NotFoundException extends NetworkException {
  /// Constructs a [NotFoundException] with the provided message and optional stack trace.
  NotFoundException(super.message, [super.stackTrace]);
}
