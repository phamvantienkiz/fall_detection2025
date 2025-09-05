import 'dart:convert';
import 'dart:io'; // Để kiểm tra Platform.isAndroid
import 'dart:typed_data'; // For Int64List

import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:vibration/vibration.dart'; // Thêm package vibration nếu muốn rung tùy chỉnh mạnh hơn

// Import GlobalKey từ main.dart
import '../main.dart'; // Điều chỉnh đường dẫn nếu cần

class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FirebaseMessaging _firebaseMessaging = FirebaseMessaging.instance;
  final FlutterLocalNotificationsPlugin _flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  static const String criticalChannelId = 'critical_fall_alert_channel';
  static const String criticalChannelName = 'Critical Fall Alerts';
  static const String criticalChannelDescription =
      'Channel for critical fall detection alerts';
  // KHÔNG CẦN ÂM THANH TÙY CHỈNH NỮA
  // static const String customSound = 'emergency_alert';

  Future<void> initialize() async {
    if (Platform.isAndroid) {
      await _createAndroidNotificationChannel();
      await _requestAndroidPermissions();
    }

    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings(
            '@mipmap/ic_launcher'); // Icon app của bạn

    // Cài đặt cho iOS (có thể bỏ qua nếu chỉ tập trung Android)
    const DarwinInitializationSettings initializationSettingsIOS =
        DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
      // onDidReceiveLocalNotification: onDidReceiveLocalNotification, // Cho iOS < 10
    );

    const InitializationSettings initializationSettings =
        InitializationSettings(
      android: initializationSettingsAndroid,
      iOS: initializationSettingsIOS, // Bỏ qua nếu chỉ Android
    );

    await _flutterLocalNotificationsPlugin.initialize(
      initializationSettings,
      onDidReceiveNotificationResponse: _onDidReceiveNotificationResponse,
      onDidReceiveBackgroundNotificationResponse:
          _onDidReceiveBackgroundNotificationResponse,
    );

    _configureFirebaseListeners();
  }

  Future<void> _requestAndroidPermissions() async {
    // Yêu cầu quyền POST_NOTIFICATIONS cho Android 13+
    // flutter_local_notifications sẽ tự động yêu cầu khi cần thiết
    // Hoặc bạn có thể dùng permission_handler
    final AndroidFlutterLocalNotificationsPlugin? androidImplementation =
        _flutterLocalNotificationsPlugin.resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>();
    if (androidImplementation != null) {
      final bool? granted =
          await androidImplementation.requestNotificationsPermission();
      print("Android Notification Permission Granted: $granted");
    }
  }

  Future<void> _createAndroidNotificationChannel() async {
    final Int64List vibrationPattern =
        Int64List.fromList([0, 1000, 500, 2000, 500, 1000]);

    final AndroidNotificationChannel channel = AndroidNotificationChannel(
      criticalChannelId,
      criticalChannelName,
      description: criticalChannelDescription,
      importance: Importance.max, // QUAN TRỌNG: Đặt mức độ ưu tiên cao nhất
      playSound: true, // QUAN TRỌNG: Bật âm thanh
      enableVibration: true,
      vibrationPattern: vibrationPattern,
      // Không cần chỉ định sound, nó sẽ dùng âm thanh mặc định của kênh
    );

    await _flutterLocalNotificationsPlugin
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(channel);
    print("Android Notification Channel Created: $criticalChannelId");
  }

  void _configureFirebaseListeners() {
    // 1. Xử lý khi app ở nền trước (foreground)
    FirebaseMessaging.onMessage.listen((RemoteMessage message) async {
      print('Foreground FCM Message Received:');
      print('Title: ${message.notification?.title}');
      print('Body: ${message.notification?.body}');
      print('Data: ${message.data}');

      // Kiểm tra nếu là thông báo FALL
      if (message.data['event'] == 'FALL') {
        // Rung thiết bị
        if (await Vibration.hasVibrator() ?? false) {
          Vibration.vibrate(
            pattern: [0, 500, 200, 800],
            intensities: [0, 255, 0, 255],
          );
        }

        // Chuyển thẳng sang màn hình alert_screen.dart
        navigatorKey.currentState?.pushNamed(
          '/alert',
          arguments: message.data, // Truyền data payload
        );
      }
      // Không hiển thị local notification ở foreground
    });

    // 2. Xử lý khi nhấn vào thông báo (app từ background/terminated mở lên)
    FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
      print('FCM Message Tapped (App was in background):');
      print('Data: ${message.data}');
      _handleNotificationTap(message.data, message.messageId);
    });

    // 3. Xử lý nếu app được mở từ trạng thái terminated bởi một thông báo
    FirebaseMessaging.instance
        .getInitialMessage()
        .then((RemoteMessage? message) {
      if (message != null) {
        print('FCM Message Tapped (App was terminated):');
        print('Data: ${message.data}');
        _handleNotificationTap(message.data, message.messageId);
      }
    });
  }

  void _handleNotificationTap(
      Map<String, dynamic> data, String? notificationId) {
    if (data['event'] == 'FALL') {
      navigatorKey.currentState?.pushNamed(
        '/alert',
        arguments: data,
      );
    }
    // Tắt tất cả thông báo khi người dùng tương tác với một thông báo
    // để dừng âm thanh/rung lặp lại.
    cancelAllNotifications();
  }

  // Hàm này được gọi từ onDidReceiveNotificationResponse và onDidReceiveBackgroundNotificationResponse
  // khi người dùng nhấn vào một LOCAL notification
  void _handleLocalNotificationTap(NotificationResponse notificationResponse) {
    print(
        'Local Notification Tapped. Payload: ${notificationResponse.payload}');

    // Dừng rung ngay lập tức
    Vibration.cancel();

    if (notificationResponse.payload != null &&
        notificationResponse.payload!.isNotEmpty) {
      try {
        final Map<String, dynamic> data =
            jsonDecode(notificationResponse.payload!);
        if (data['event'] == 'FALL') {
          navigatorKey.currentState?.pushNamed(
            '/alert',
            arguments: data,
          );
        }
      } catch (e) {
        print("Error decoding local notification payload: $e");
      }
    }
    // Tắt tất cả thông báo để đảm bảo thông báo đang hoạt động (ongoing) bị hủy
    cancelAllNotifications();
  }

  // Hàm này được gọi khi người dùng nhấn vào một local notification
  static void _onDidReceiveNotificationResponse(
      NotificationResponse notificationResponse) {
    NotificationService()._handleLocalNotificationTap(notificationResponse);
  }

  // Hàm này được gọi khi người dùng nhấn vào một local notification từ background
  @pragma('vm:entry-point')
  static void _onDidReceiveBackgroundNotificationResponse(
      NotificationResponse notificationResponse) {
    // Xử lý logic nền nếu cần, hoặc chỉ đơn giản là chuẩn bị dữ liệu
    // Việc điều hướng sẽ xảy ra khi app được đưa lên foreground
    print(
        'Background Local Notification Tapped. Payload: ${notificationResponse.payload}');
    // Không nên điều hướng trực tiếp từ đây vì app có thể chưa sẵn sàng.
    // onMessageOpenedApp hoặc getInitialMessage sẽ xử lý khi app mở.
    // Tuy nhiên, nếu bạn muốn lưu trữ payload để xử lý sau, có thể làm ở đây.
  }

  // Hàm này được gọi bởi _firebaseMessagingBackgroundHandler
  // để hiển thị thông báo popup bền bỉ khi app ở background/terminated
  Future<void> showPersistentFallNotification(RemoteMessage message) async {
    print("Showing persistent local notification from background handler.");

    final Map<String, dynamic> data = message.data;
    final String title =
        data['title'] ?? message.notification?.title ?? "CẢNH BÁO NGÃ!";
    final String body = data['body'] ??
        message.notification?.body ??
        "Phát hiện có người bị ngã. Hãy kiểm tra ngay!";

    // Lặp lại rung và âm thanh
    final Int64List vibrationPattern =
        Int64List.fromList([0, 1000, 1000, 1000, 1000, 1000]);

    final AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
      criticalChannelId,
      criticalChannelName,
      channelDescription: criticalChannelDescription,
      importance: Importance.max,
      priority: Priority.high,
      playSound: true, // Bật âm thanh
      enableVibration: true,
      vibrationPattern: vibrationPattern,
      ongoing: true, // QUAN TRỌNG: Thông báo sẽ không tự biến mất
      autoCancel: false, // QUAN TRỌNG: Thông báo không hủy khi nhấn
      ticker: 'Cảnh báo ngã!',
    );

    const DarwinNotificationDetails iOSPlatformChannelSpecifics =
        DarwinNotificationDetails(
      presentSound: true, // Bật âm thanh cho iOS
      // critical: true, // Cân nhắc dùng cho iOS 15+
    );

    final NotificationDetails platformChannelSpecifics = NotificationDetails(
      android: androidPlatformChannelSpecifics,
      iOS: iOSPlatformChannelSpecifics,
    );

    // Sử dụng hashCode của messageId hoặc thời gian để có ID duy nhất
    final int notificationId = message.messageId?.hashCode ??
        DateTime.now().millisecondsSinceEpoch % 100000;

    await _flutterLocalNotificationsPlugin.show(
      notificationId,
      title,
      body,
      platformChannelSpecifics,
      payload: jsonEncode(data), // Truyền data để xử lý khi nhấn
    );
    print("Persistent fall notification shown with ID: $notificationId");
  }

  // Hàm để hủy tất cả thông báo và dừng rung
  Future<void> cancelAllNotifications() async {
    await _flutterLocalNotificationsPlugin.cancelAll();
    if (await Vibration.hasVibrator() ?? false) {
      Vibration.cancel();
    }
    print("All notifications and vibrations cancelled.");
  }
}
