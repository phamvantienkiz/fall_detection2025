import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';

class FCMService {
  final FirebaseMessaging _firebaseMessaging = FirebaseMessaging.instance;

  Future<String?> getFCMToken() async {
    try {
      // Request permission for iOS and web
      NotificationSettings settings =
          await _firebaseMessaging.requestPermission(
        alert: true,
        announcement: false,
        badge: true,
        carPlay: false,
        criticalAlert: false,
        provisional: false,
        sound: true,
      );

      if (kDebugMode) {
        print('User granted permission: \${settings.authorizationStatus}');
      }

      if (settings.authorizationStatus == AuthorizationStatus.authorized) {
        String? token = await _firebaseMessaging.getToken();
        if (kDebugMode) {
          print('FCM Token: \$token');
        }
        return token;
      } else {
        if (kDebugMode) {
          print('User declined or has not accepted permission');
        }
        return null;
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error getting FCM token: \$e');
      }
      return null;
    }
  }
}
