import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';

export default function AboutScreen() {
  const router = useRouter();

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.pageTitle}>About Our System</Text>

      <View style={styles.card}>
        <Text style={styles.cardIcon}>🫀</Text>
        <Text style={styles.cardTitle}>Heart Disease Identification System</Text>
        <Text style={styles.cardText}>
          Our Heart Disease Identification System is designed to aid doctors, medical professionals, and patients in detecting cardiovascular conditions accurately, minimizing risks, and improving health outcomes.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardIcon}>🧠</Text>
        <Text style={styles.cardTitle}>Our Technology</Text>
        <Text style={styles.cardText}>
          We utilize cutting-edge deep learning techniques combined with medical data analysis to ensure fast and accurate heart disease identification through key health indicators and predictive analytics.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardIcon}>📊</Text>
        <Text style={styles.cardTitle}>Advanced Services</Text>
        <Text style={styles.cardText}>
          Our system uses multiple ML algorithms including SVM, Decision Trees, ANN, and HMM to provide the most accurate predictions. The best performing model is automatically selected for every prediction.
        </Text>
      </View>

      <View style={styles.featuresRow}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>4+</Text>
          <Text style={styles.statLabel}>ML Models</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>95%+</Text>
          <Text style={styles.statLabel}>Accuracy</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>11</Text>
          <Text style={styles.statLabel}>Health Indicators</Text>
        </View>
      </View>

      <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
        <Text style={styles.backText}>← Back to Home</Text>
      </TouchableOpacity>

      <Text style={styles.footer}>© 2026 Heart Disease Prediction System | MSL Corporations</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f4f7f6' },
  content: { padding: 20, paddingTop: 50, paddingBottom: 40 },
  pageTitle: { fontSize: 26, fontWeight: '800', color: '#1e3c72', textAlign: 'center', marginBottom: 20 },
  card: {
    backgroundColor: '#fff', borderRadius: 18, padding: 22, marginBottom: 15,
    shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.08, shadowRadius: 10, elevation: 3,
    borderLeftWidth: 4, borderLeftColor: '#FF6B6B',
  },
  cardIcon: { fontSize: 35, marginBottom: 8 },
  cardTitle: { fontSize: 17, fontWeight: '700', color: '#2C3E50', marginBottom: 8 },
  cardText: { fontSize: 13, color: '#666', lineHeight: 20 },
  featuresRow: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 5, marginBottom: 20 },
  statCard: {
    backgroundColor: '#1e3c72', borderRadius: 14, padding: 18, width: '31%', alignItems: 'center',
    shadowColor: '#1e3c72', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8, elevation: 4,
  },
  statNumber: { fontSize: 22, fontWeight: '900', color: '#FF6B6B' },
  statLabel: { fontSize: 10, fontWeight: '600', color: 'rgba(255,255,255,0.8)', marginTop: 3, textAlign: 'center' },
  backBtn: { alignItems: 'center', marginTop: 5 },
  backText: { color: '#2a5298', fontSize: 14, fontWeight: '600' },
  footer: { textAlign: 'center', color: '#aaa', fontSize: 10, marginTop: 20 },
});
