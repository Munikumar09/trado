import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/utils/validators.dart';

void main() {
  group('Validators.email', () {
    test('should return error when email is null', () {
      final result = Validators.email(null);
      expect(result, 'Please enter your email');
    });

    test('should return error when email is empty', () {
      final result = Validators.email('');
      expect(result, 'Please enter your email');
    });

    test('should return error when email has multiple @ symbols', () {
      final result = Validators.email('test@@example.com');
      expect(result, 'Please enter a valid email address');
    });

    test('should return error when email has an invalid TLD', () {
      final result = Validators.email('test@example.');
      expect(result, 'Please enter a valid email address');
    });
    test('should return error when email contains invalid special characters',
        () {
      final result = Validators.email('test!@example.com');
      expect(result, 'Please enter a valid email address');
    });

    test('should return error for an invalid email format', () {
      final result = Validators.email('invalidemail');
      expect(result, 'Please enter a valid email address');
    });

    test('should return null when email is valid', () {
      final result = Validators.email('test@example.com');
      expect(result, null);
    });
    test('should return null when email with subdomain is valid', () {
      final result = Validators.email('test@sub.example.com');
      expect(result, null);
    });
  });

  group('Validators.password', () {
    test('should return error when password is null', () {
      final result = Validators.password(null);
      expect(result, 'Please enter your password');
    });

    test('should return error when password is empty', () {
      final result = Validators.password('');
      expect(result, 'Please enter your password');
    });

    test('should return error when password is less than 8 characters', () {
      final result = Validators.password('Ab1!');
      expect(result, 'Password must be at least 8 characters long');
    });

    test('should return error when password has no uppercase letter', () {
      final result = Validators.password('abcdefg1!');
      expect(result, 'Password must contain at least one uppercase letter');
    });

    test('should return error when password has no number', () {
      final result = Validators.password('Abcdefgh!');
      expect(result, 'Password must contain at least one number');
    });

    test('should return error when password has no special character', () {
      final result = Validators.password('Abcdefg1');
      expect(result, 'Password must contain at least one special character');
    });

    test('should return null when password is valid', () {
      final result = Validators.password('Abcdef1!');
      expect(result, null);
    });
  });

  group('Validators.required', () {
    test('should return error when value is null', () {
      final result = Validators.required(null);
      expect(result, 'This field is required');
    });

    test('should return error when value is empty', () {
      final result = Validators.required('');
      expect(result, 'This field is required');
    });

    test('should return null when value is provided', () {
      final result = Validators.required('Some value');
      expect(result, null);
    });
  });

  group('Validators.confirmPassword', () {
    test('should return error when confirmPassword is null', () {
      final result = Validators.confirmPassword('Password1!', null);
      expect(result, 'Please confirm your password');
    });

    test('should return error when confirmPassword is empty', () {
      final result = Validators.confirmPassword('Password1!', '');
      expect(result, 'Please confirm your password');
    });

    test('should return error when original password is null', () {
      final resultNull = Validators.confirmPassword(null, 'Password1!');
      expect(resultNull, 'Passwords do not match');
    });
    test('should return error when original password is empty', () {
      final resultEmpty = Validators.confirmPassword('', 'Password1!');
      expect(resultEmpty, 'Passwords do not match');
    });

    test('should return error when passwords do not match', () {
      final result = Validators.confirmPassword('Password1!', 'Password2@');
      expect(result, 'Passwords do not match');
    });

    test('should return null when passwords match', () {
      final result = Validators.confirmPassword('Password1!', 'Password1!');
      expect(result, null);
    });
  });

  group('Validators.phoneNumber', () {
    test('should return error when phone number is null', () {
      final result = Validators.phoneNumber(null);
      expect(result, 'Please enter your phone number');
    });

    test('should return error when phone number is empty', () {
      final result = Validators.phoneNumber('');
      expect(result, 'Please enter your phone number');
    });

    test('should return an error for a phone number with not exactly 10 digits',
        () {
      final result = Validators.phoneNumber('123456789');
      expect(result, 'Please enter a valid phone number');
    });

    // If your validator ONLY accepts 10 digits:
    test('should return error for an invalid phone number', () {
      final result = Validators.phoneNumber('123-abc');
      expect(result, 'Please enter a valid phone number');
    });

    test('should return error for a phone number with spaces', () {
      final result = Validators.phoneNumber('123 456 7890');
      expect(result, 'Please enter a valid phone number');
    });

    test('should return error for a phone number with dashes', () {
      final result = Validators.phoneNumber('123-456-7890');
      expect(result, 'Please enter a valid phone number');
    });

    test('should return error for an international format phone number', () {
      final result = Validators.phoneNumber('+1-234-567-890');
      expect(result, 'Please enter a valid phone number');
    });

    test(
        'should return error when phone number with country code and  is valid',
        () {
      final result = Validators.phoneNumber('+11234567890');
      expect(result, "Please enter a valid phone number");
    });

    test('should return null when phone number is valid', () {
      final result = Validators.phoneNumber('1234567890');
      expect(result, null);
    });
  });
}
