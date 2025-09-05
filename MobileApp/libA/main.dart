// ignore_for_file: unused_import

import 'package:flutter/material.dart';
import 'package:fall_tracking/screen/home_screen.dart';
import 'package:fall_tracking/screen/alert_screen.dart';
import 'package:fall_tracking/services/notification_service.dart';
import 'package:firebase_core/firebase_core.dart'; // Import Firebase Core
import '../../Mobile/lib/firebase_options.dart'; // Import generated firebase_options.dart
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:fall_tracking/screen/listFall_screen.dart'; // Import ListFallScreen

// Khai báo GlobalKey cho Navigator
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

// Background message handler (phải là top-level function)
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // Quan trọng: Khởi tạo Firebase nếu bạn cần dùng các dịch vụ Firebase khác ở đây
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  print("Handling a background message: ${message.messageId}");
  print("Background Message data: ${message.data}");

  if (message.data['event'] == 'FALL') {
    // Khởi tạo một instance mới của service để gọi hàm
    await NotificationService().initialize(); // Đảm bảo service được khởi tạo
    await NotificationService().showPersistentFallNotification(message);
  }
}

void main() async {
  // Make main async
  WidgetsFlutterBinding
      .ensureInitialized(); // Ensure Flutter bindings are initialized
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform, // Initialize Firebase
  );

  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

  // Khởi tạo NotificationService
  await NotificationService().initialize();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Fall Tracking App',
      navigatorKey: navigatorKey,
      theme: ThemeData(
        primarySwatch: Colors.purple,
        fontFamily: 'Poppins',
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const FallTrackingHomeScreen(),
        '/alert': (context) {
          final args = ModalRoute.of(context)!.settings.arguments
              as Map<String, dynamic>?;
          return FallAlertScreen(payload: args ?? {});
        },
        '/listFall': (context) => const ListFallScreen(),
      },
    );
  }
}
