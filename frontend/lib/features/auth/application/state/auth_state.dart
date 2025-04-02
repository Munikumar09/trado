/*
Documentation:
---------------
File: auth_state.dart
Description:
  This file defines the authentication state management using an immutable AuthState class and an AuthStatus enum.
  The AuthStatus enum represents various stages of authentication.
  AuthState provides several factory constructors for creating different authentication states and a copyWith method for state updates.
  
Methods:
  • AuthState.initial():
      - Returns the default initial authentication state.
  • AuthState.authenticated(String accessToken):
      - Creates an authenticated state with the provided access token.
  • AuthState.unverified({required String email, String? error}):
      - Creates a state for an unverified user, optionally including an error message.
  • AuthState.unauthenticated():
      - Creates a state representing an unauthenticated user.
  • AuthState.error(String error):
      - Creates a state representing an error with an associated message.
  • copyWith({…}):
      - Returns a new AuthState instance with modified properties.
  • toString():
      - Returns a string representation of AuthState.
*/

enum AuthStatus {
  initial,
  loading,
  authenticated,
  unauthenticated,
  verificationPending,
  verificationSent,
  unverified,
  verified,
  error
}

class AuthState {
  final AuthStatus status;
  final String? email;
  final String? error;
  final String? accessToken;

  const AuthState._({
    required this.status,
    this.email,
    this.error,
    this.accessToken,
  });

  /// Returns a new AuthState representing the initial state.
  factory AuthState.initial() =>
      const AuthState._(status: AuthStatus.initial, error: null);

  /// Creates an authenticated AuthState with the given access token.
  factory AuthState.authenticated(String accessToken) =>
      AuthState._(status: AuthStatus.authenticated, accessToken: accessToken);

  /// Creates an unverified AuthState with the provided email and optional error message.
  factory AuthState.unverified({required String email, String? error}) =>
      AuthState._(status: AuthStatus.unverified, email: email, error: error);

  /// Returns an unauthenticated AuthState.
  factory AuthState.unauthenticated() =>
      const AuthState._(status: AuthStatus.unauthenticated);

  /// Creates an error AuthState with the given error message.
  factory AuthState.error(String error) =>
      AuthState._(status: AuthStatus.error, error: error);

  /// Creates a new AuthState instance with updated properties.
  AuthState copyWith({
    AuthStatus? status,
    String? email,
    String? error,
    String? accessToken,
  }) {
    return AuthState._(
      status: status ?? this.status,
      email: email ?? this.email,
      error: error ?? this.error,
      accessToken: accessToken ?? this.accessToken,
    );
  }

  /// Returns a string representation of the AuthState.
  @override
  String toString() {
    return 'AuthState(status: $status, email: $email, error: $error, accessToken: ${accessToken != null ? '***' : 'null'})';
  }
}
