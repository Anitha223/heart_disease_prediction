import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Animated } from 'react-native';
import { useRouter } from 'expo-router';

export default function HomeScreen() {
  const router = useRouter();
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 800, useNativeDriver: true }).start();
  }, []);

  const menuItems = [
    { icon: '💊', title: 'Prediction', desc: 'Predict heart disease risk', route: '/(tabs)/predict', color: '#FF6B6B' },
    { icon: '📈', title: 'Classification View', desc: 'View Model Accuracies', route: '/(tabs)/classification', color: '#27ae60' },
    { icon: '📊', title: 'About System', desc: 'Our technology & services', route: '/(tabs)/about', color: '#2a5298' },
  ];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View style={{ opacity: fadeAnim }}>
        {/* Header */}
        <View style={styles.headerCard}>
          <Text style={styles.welcomeText}>Welcome, User</Text>
          <Text style={styles.welcomeSub}>Utilize our cutting-edge Heart Disease Prediction System to accurately assess cardiovascular risks using our advanced Machine Learning features.</Text>
        </View>

        {/* Features */}
        <Text style={styles.sectionTitle}>Your Trusted Partner for Accurate Heart Disease Detection</Text>
        <Text style={styles.sectionSub}>We are dedicated to providing innovative solutions that enhance early diagnosis and preventive healthcare.</Text>

        <View style={styles.featuresRow}>
          <View style={styles.featureCard}>
            <Text style={styles.featureIcon}>💓</Text>
            <Text style={styles.featureTitle}>Accurate Detection</Text>
            <Text style={styles.featureDesc}>Advanced ML to analyze medical data with high precision</Text>
          </View>
          <View style={styles.featureCard}>
            <Text style={styles.featureIcon}>⚡</Text>
            <Text style={styles.featureTitle}>Real-Time Analysis</Text>
            <Text style={styles.featureDesc}>Instant results powered by AI for fast prediction</Text>
          </View>
          <View style={styles.featureCard}>
            <Text style={styles.featureIcon}>👨‍⚕️</Text>
            <Text style={styles.featureTitle}>Patient Care</Text>
            <Text style={styles.featureDesc}>Supports healthcare providers in informed decisions</Text>
          </View>
        </View>

        {/* Menu Cards */}
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        {menuItems.map((item, index) => (
          <TouchableOpacity key={index} style={[styles.menuCard, { borderLeftColor: item.color }]} onPress={() => router.push(item.route as any)}>
            <Text style={styles.menuIcon}>{item.icon}</Text>
            <View style={styles.menuTextWrap}>
              <Text style={styles.menuTitle}>{item.title}</Text>
              <Text style={styles.menuDesc}>{item.desc}</Text>
            </View>
            <Text style={styles.menuArrow}>→</Text>
          </TouchableOpacity>
        ))}

        {/* Logout */}
        <TouchableOpacity style={styles.logoutBtn} onPress={() => router.replace('/(tabs)/')}>
          <Text style={styles.logoutText}>🚪 Logout</Text>
        </TouchableOpacity>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f4f7f6' },
  content: { padding: 20, paddingTop: 50, paddingBottom: 40 },
  headerCard: {
    backgroundColor: '#FF6B6B', borderRadius: 20, padding: 25, marginBottom: 25,
    shadowColor: '#FF6B6B', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.3, shadowRadius: 15, elevation: 6,
  },
  welcomeText: { fontSize: 24, fontWeight: '800', color: '#fff' },
  welcomeSub: { fontSize: 13, color: 'rgba(255,255,255,0.85)', marginTop: 8, lineHeight: 20 },
  sectionTitle: { fontSize: 18, fontWeight: '700', color: '#2C3E50', marginBottom: 5, marginTop: 10 },
  sectionSub: { fontSize: 12, color: '#888', marginBottom: 15 },
  featuresRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 },
  featureCard: {
    backgroundColor: '#fff', borderRadius: 14, padding: 12, width: '31%', alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
  },
  featureIcon: { fontSize: 28, marginBottom: 6 },
  featureTitle: { fontSize: 11, fontWeight: '700', color: '#2C3E50', textAlign: 'center' },
  featureDesc: { fontSize: 9, color: '#888', textAlign: 'center', marginTop: 3 },
  menuCard: {
    backgroundColor: '#fff', borderRadius: 15, padding: 18, flexDirection: 'row', alignItems: 'center',
    marginBottom: 12, borderLeftWidth: 4,
    shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
  },
  menuIcon: { fontSize: 32, marginRight: 15 },
  menuTextWrap: { flex: 1 },
  menuTitle: { fontSize: 16, fontWeight: '700', color: '#2C3E50' },
  menuDesc: { fontSize: 12, color: '#888', marginTop: 2 },
  menuArrow: { fontSize: 20, color: '#ccc' },
  logoutBtn: {
    borderWidth: 2, borderColor: '#e74c3c', borderRadius: 30, padding: 14, alignItems: 'center', marginTop: 15,
  },
  logoutText: { color: '#e74c3c', fontSize: 15, fontWeight: '700' },
});
