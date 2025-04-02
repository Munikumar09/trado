/*
Documentation:
---------------
Class: SignupRequest
Description:
  Represents a user signup request containing necessary details for registration.
  Provides a method to convert its data into JSON format for API submission.

Methods:
  • toJson():
      - Converts the SignupRequest instance into a JSON map.
      - Example: final jsonData = signupRequest.toJson();
*/

// Code:
/// Represents a user signup request containing necessary details for registration.
class SignupRequest {
  final String email;
  final String username;
  final String password;
  final String confirmPassword;
  final String dateOfBirth;
  final String phoneNumber;
  final String gender;

  SignupRequest({
    required this.email,
    required this.username,
    required this.password,
    required this.confirmPassword,
    required this.dateOfBirth,
    required this.phoneNumber,
    required this.gender,
  });

  /// Converts the signup request fields into a JSON map.
  Map<String, dynamic> toJson() => {
        'email': email,
        'username': username,
        'password': password,
        'confirm_password': confirmPassword,
        'date_of_birth': dateOfBirth,
        'phone_number': phoneNumber,
        'gender': gender,
      };
}
