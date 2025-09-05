import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // Added for Clipboard

class TokenDetailsScreen extends StatelessWidget {
  final String tokenCode; // This will now be the FCM token

  final String fcmToken = "cpa...";

  const TokenDetailsScreen({
    super.key,
    required this.tokenCode, // Changed to tokenCode, which is the FCM token
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // AppBar được đặt bên trong một Container có gradient để đồng bộ với body
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title:
            const Text('Token Details', style: TextStyle(color: Colors.white)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Container(
        // Gradient nền
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF6A1B9A), Color(0xFF8E24AA)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0),
            child: _buildDetailsCard(context),
          ),
        ),
      ),
    );
  }

  // Widget cho thẻ thông tin màu trắng
  Widget _buildDetailsCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16.0),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min, // Giúp thẻ co lại vừa với nội dung
        children: [
          _buildFcmTokenSection(),
          const SizedBox(height: 32),
          _buildCopyButton(context), // Changed to _buildCopyButton
        ],
      ),
    );
  }

  // Phần hiển thị FCM Token có thể copy
  Widget _buildFcmTokenSection() {
    return Column(
      children: [
        const Text(
          'Your FCM Token', // Simplified text
          style: TextStyle(color: Colors.grey, fontSize: 14),
        ),
        const SizedBox(height: 8),
        SelectableText(
          tokenCode, // Use tokenCode (FCM token) passed to the screen
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.black87,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  // Widget cho nút "Copy"
  Widget _buildCopyButton(BuildContext context) {
    // Renamed from _buildContinueButton
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: () {
          Clipboard.setData(ClipboardData(text: tokenCode)); // Copy FCM token
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('FCM Token Copied to Clipboard')),
          );
          // Không pop màn hình ở đây
        },
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF4CAF50),
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
        child: const Text(
          'Copy', // Changed text to Copy
          style: TextStyle(fontSize: 16, color: Colors.white),
        ),
      ),
    );
  }
}
