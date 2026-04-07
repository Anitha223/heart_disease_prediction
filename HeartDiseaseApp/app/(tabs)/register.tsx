import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { API_BASE } from '../config';

export default function RegisterScreen() {
  const router = useRouter();
  const [form, setForm] = useState({
    name: '', loginid: '', email: '', password: '', mobile: '', locality: '', state: '',
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (key: string, value: string) => setForm({ ...form, [key]: value });

  const handleRegister = async () => {
    const emptyField = Object.entries(form).find(([_, v]) => !v);
    if (emptyField) {
      Alert.alert('Error', `Please fill in ${emptyField[0].replace(/_/g, ' ')}`);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (data.success) {
        Alert.alert('Success', data.message, [{ text: 'OK', onPress: () => router.push('/(tabs)/login') }]);
      } else {
        Alert.alert('Registration Failed', data.error || 'Please check your details');
      }
    } catch (error) {
      Alert.alert('Connection Error', 'Ensure Django server and localtunnel are running.');
    } finally {
      setLoading(false);
    }
  };

  const fields = [
    { key: 'name', label: 'Full Name', placeholder: 'Enter your name', keyboard: 'default' as const },
    { key: 'loginid', label: 'Login ID', placeholder: 'Choose a login ID (letters only, 5-20)', keyboard: 'default' as const },
    { key: 'email', label: 'Email', placeholder: 'Enter your email', keyboard: 'email-address' as const },
    { key: 'password', label: 'Password', placeholder: 'Min 8 chars (letters + numbers)', keyboard: 'default' as const, secure: true },
    { key: 'mobile', label: 'Mobile', placeholder: '10-digit mobile number', keyboard: 'phone-pad' as const },
    { key: 'locality', label: 'Locality', placeholder: 'Enter your locality', keyboard: 'default' as const },
    { key: 'state', label: 'State', placeholder: 'Enter your state', keyboard: 'default' as const },
  ];

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Create Account</Text>
        <Text style={styles.subtitle}>Register for Heart Disease Identification</Text>

        <View style={styles.card}>
          {fields.map((f) => (
            <View key={f.key}>
              <Text style={styles.label}>{f.label.toUpperCase()}</Text>
              <TextInput
                style={styles.input}
                value={(form as any)[f.key]}
                onChangeText={(val) => handleChange(f.key, val)}
                placeholder={f.placeholder}
                placeholderTextColor="#999"
                keyboardType={f.keyboard}
                secureTextEntry={f.secure || false}
                autoCapitalize="none"
              />
            </View>
          ))}

          <TouchableOpacity style={styles.button} onPress={handleRegister} disabled={loading}>
            {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>REGISTER</Text>}
          </TouchableOpacity>

          <TouchableOpacity onPress={() => router.push('/(tabs)/login')}>
            <Text style={styles.linkText}>Already have an account? <Text style={styles.linkBold}>Login here</Text></Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity onPress={() => router.back()}>
          <Text style={styles.backText}>← Back to Welcome</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1e3c72' },
  scroll: { flexGrow: 1, padding: 25, paddingTop: 50, paddingBottom: 40 },
  title: { fontSize: 26, fontWeight: '800', color: '#fff', textAlign: 'center' },
  subtitle: { fontSize: 13, color: 'rgba(255,255,255,0.6)', textAlign: 'center', marginBottom: 20 },
  card: {
    backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: 20, padding: 22,
    shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.15, shadowRadius: 20, elevation: 8,
  },
  label: { fontSize: 11, fontWeight: '700', color: '#2a5298', marginBottom: 4, marginTop: 10 },
  input: {
    borderWidth: 1, borderColor: '#ddd', borderRadius: 10, padding: 12, fontSize: 14,
    backgroundColor: '#f8f9fa', color: '#333',
  },
  button: {
    backgroundColor: '#FF6B6B', padding: 16, borderRadius: 30, alignItems: 'center', marginTop: 18,
    shadowColor: '#e74c3c', shadowOffset: { width: 0, height: 5 }, shadowOpacity: 0.3, shadowRadius: 10, elevation: 5,
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '700', letterSpacing: 1.5 },
  linkText: { textAlign: 'center', marginTop: 15, color: '#666', fontSize: 13 },
  linkBold: { color: '#2a5298', fontWeight: '700' },
  backText: { textAlign: 'center', color: 'rgba(255,255,255,0.7)', marginTop: 20, fontSize: 14 },
});
