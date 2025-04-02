/*
Documentation:
---------------
Class: StorageKeys
Description:
  Contains constant keys used to extract data (e.g. tokens) from response data received from the backend.
  
Fields:
  • accessToken:
      - Key used to extract the access token from the response.
      - Example: Use StorageKeys.accessToken when accessing the token from a backend response.
      
  • refreshToken:
      - Key used to extract the refresh token from the response.
      - Example: Use StorageKeys.refreshToken when accessing the token from a backend response.
*/

// Code:
/// Contains constant keys used to extract data from response data received from the backend.
abstract class StorageKeys {
  static const accessToken = 'access_token';
  static const refreshToken = 'refresh_token';
}
