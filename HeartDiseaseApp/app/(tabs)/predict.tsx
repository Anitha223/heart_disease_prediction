import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator, Alert, Animated } from 'react-native';
import { useRouter } from 'expo-router';
import { API_BASE } from '../config';

interface PredictionResult {
  prediction: string;
  risk_score_percentage: number;
  success: boolean;
}

export default function PredictScreen() {
  const router = useRouter();
  const [formData, setFormData] = useState<Record<string, string>>({
    age: '', sex: '', chest_pain_type: '', resting_bp_s: '', cholesterol: '',
    fasting_blood_sugar: '', resting_ecg: '', max_heart_rate: '', exercise_angina: '',
    oldpeak: '', st_slope: ''
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const barWidth = React.useRef(new Animated.Value(0)).current;

  const handleChange = (key: string, value: string) => setFormData({ ...formData, [key]: value });

  const handlePredict = async () => {
    const emptyField = Object.entries(formData).find(([_, v]) => !v);
    if (emptyField) {
      Alert.alert('Error', `Please fill in ${emptyField[0].replace(/_/g, ' ')}`);
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'bypass-tunnel-reminder': 'true',
        },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      if (data.success) {
        setResult(data);
        // Animate risk bar
        barWidth.setValue(0);
        Animated.timing(barWidth, {
          toValue: data.risk_score_percentage,
          duration: 1500,
          useNativeDriver: false,
        }).start();
      } else {
        Alert.alert('Error', data.error || 'Prediction failed');
      }
    } catch (error) {
      Alert.alert('Connection Error', 'Ensure Django server and localtunnel are running.');
    } finally {
      setLoading(false);
    }
  };

  const fieldLabels: Record<string, string> = {
    age: 'Age', sex: 'Gender (0=Female, 1=Male)', chest_pain_type: 'Chest Pain Type (1-4)',
    resting_bp_s: 'Resting Blood Pressure (mm Hg)', cholesterol: 'Cholesterol (mg/dl)',
    fasting_blood_sugar: 'Fasting Blood Sugar (>120:1, else:0)', resting_ecg: 'Resting ECG (0-2)',
    max_heart_rate: 'Max Heart Rate Achieved', exercise_angina: 'Exercise Angina (1=Yes, 0=No)',
    oldpeak: 'Oldpeak (ST Depression, 0-2)', st_slope: 'ST Slope (0-2)',
  };

  const riskColor = result && result.risk_score_percentage > 50 ? '#e74c3c' : '#2ecc71';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      {/* Result Section */}
      {result && (
        <View style={[styles.resultBox, { borderColor: riskColor }]}>
          <Text style={styles.resultLabel}>AI Predicted Risk</Text>
          <Text style={[styles.resultValue, { color: riskColor }]}>{result.prediction}</Text>
          
          <Text style={styles.riskMeterLabel}>Detailed Risk Meter: {result.risk_score_percentage}%</Text>
          <View style={styles.riskBarBg}>
            <Animated.View style={[styles.riskBarFill, {
              width: barWidth.interpolate({
                inputRange: [0, 100],
                outputRange: ['0%', '100%'],
              }),
              backgroundColor: riskColor,
            }]} />
          </View>
          <Text style={[styles.riskPercent, { color: riskColor }]}>{result.risk_score_percentage}%</Text>
        </View>
      )}

      {/* Form */}
      <View style={styles.card}>
        <Text style={styles.title}>Heart Disease Risk Prediction</Text>
        <Text style={styles.subtitle}>Enter the required details to predict your heart disease risk</Text>

        {Object.keys(formData).map((key) => (
          <View key={key} style={styles.inputGroup}>
            <Text style={styles.label}>{fieldLabels[key] || key}</Text>
            <TextInput
              style={styles.input}
              value={formData[key]}
              onChangeText={(val) => handleChange(key, val)}
              keyboardType="numeric"
              placeholder="Enter value"
              placeholderTextColor="#bbb"
            />
          </View>
        ))}

        <TouchableOpacity style={styles.button} onPress={handlePredict} disabled={loading}>
          {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>PREDICT RISK</Text>}
        </TouchableOpacity>
      </View>

      <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
        <Text style={styles.backText}>← Back to Home</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f4f7f6' },
  content: { padding: 20, paddingTop: 50, paddingBottom: 40 },
  card: {
    backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: 20, padding: 25,
    shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.1, shadowRadius: 15, elevation: 5,
  },
  title: { fontSize: 22, fontWeight: '800', color: '#1e3c72', textAlign: 'center' },
  subtitle: { fontSize: 12, color: '#888', textAlign: 'center', marginBottom: 15 },
  inputGroup: { marginBottom: 12 },
  label: { fontSize: 11, fontWeight: '700', color: '#2a5298', marginBottom: 4 },
  input: {
    borderWidth: 1, borderColor: '#ddd', borderRadius: 10, padding: 12, fontSize: 15,
    backgroundColor: '#fff', color: '#333',
  },
  button: {
    backgroundColor: '#FF6B6B', padding: 16, borderRadius: 30, alignItems: 'center', marginTop: 15,
    shadowColor: '#e74c3c', shadowOffset: { width: 0, height: 5 }, shadowOpacity: 0.3, shadowRadius: 10, elevation: 5,
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '700', letterSpacing: 1 },
  resultBox: {
    backgroundColor: '#fff', borderRadius: 18, padding: 20, marginBottom: 20,
    borderWidth: 2, alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 5 }, shadowOpacity: 0.1, shadowRadius: 10, elevation: 4,
  },
  resultLabel: { fontSize: 14, color: '#666', fontWeight: '600' },
  resultValue: { fontSize: 28, fontWeight: '900', marginTop: 5 },
  riskMeterLabel: { fontSize: 13, color: '#34495e', fontWeight: '600', marginTop: 15, alignSelf: 'flex-start' },
  riskBarBg: {
    width: '100%', height: 24, backgroundColor: '#ecf0f1', borderRadius: 12, marginTop: 8, overflow: 'hidden',
  },
  riskBarFill: { height: '100%', borderRadius: 12 },
  riskPercent: { fontSize: 22, fontWeight: '900', marginTop: 5 },
  backBtn: { marginTop: 15, alignItems: 'center' },
  backText: { color: '#2a5298', fontSize: 14, fontWeight: '600' },
});
