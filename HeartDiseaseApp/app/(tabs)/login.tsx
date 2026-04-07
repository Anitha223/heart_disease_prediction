import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { API_BASE } from '../config';

export default function LoginScreen() {
  const router = useRouter();
  const [loginid, setLoginid] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!loginid || !password) {
      Alert.alert('Error', 'Please enter Login ID and Password');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/login`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'bypass-tunnel-reminder': 'true',
        },
        body: JSON.stringify({ loginid, password }),
      });
      const data = await response.json();
      if (data.success) {
        Alert.alert('Success', `Welcome ${data.name}!`);
        router.replace('/(tabs)/home');
      } else {
        Alert.alert('Login Failed', data.error || 'Invalid credentials');
      }
    } catch (error) {
      Alert.alert('Connection Error', 'Ensure Django server and localtunnel are running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.heartIcon}>❤️</Text>
        <Text style={styles.title}>User Login</Text>
        <Text style={styles.subtitle}>Heart Disease Identification System</Text>

        <View style={styles.card}>
          <Text style={styles.label}>LOGIN ID</Text>
          <TextInput style={styles.input} value={loginid} onChangeText={setLoginid} placeholder="Enter your Login ID" placeholderTextColor="#999" autoCapitalize="none" />

          <Text style={styles.label}>PASSWORD</Text>
          <TextInput style={styles.input} value={password} onChangeText={setPassword} placeholder="Enter your password" placeholderTextColor="#999" secureTextEntry />

          <TouchableOpacity style={styles.button} onPress={handleLogin} disabled={loading}>
            {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>LOGIN</Text>}
          </TouchableOpacity>

          <TouchableOpacity onPress={() => router.push('/(tabs)/register')}>
            <Text style={styles.linkText}>Don't have an account? <Text style={styles.linkBold}>Register here</Text></Text>
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
  scroll: { flexGrow: 1, justifyContent: 'center', padding: 25 },
  heartIcon: { fontSize: 50, textAlign: 'center', marginBottom: 10 },
  title: { fontSize: 28, fontWeight: '800', color: '#fff', textAlign: 'center' },
  subtitle: { fontSize: 13, color: 'rgba(255,255,255,0.6)', textAlign: 'center', marginBottom: 30 },
  card: {
    backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: 20, padding: 25,
    shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.15, shadowRadius: 20, elevation: 8,
  },
  label: { fontSize: 12, fontWeight: '700', color: '#2a5298', marginBottom: 6, marginTop: 12 },
  input: {
    borderWidth: 1, borderColor: '#ddd', borderRadius: 12, padding: 14, fontSize: 15,
    backgroundColor: '#f8f9fa', color: '#333',
  },
  button: {
    backgroundColor: '#FF6B6B', padding: 16, borderRadius: 30, alignItems: 'center', marginTop: 20,
    shadowColor: '#e74c3c', shadowOffset: { width: 0, height: 5 }, shadowOpacity: 0.3, shadowRadius: 10, elevation: 5,
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '700', letterSpacing: 1.5 },
  linkText: { textAlign: 'center', marginTop: 18, color: '#666', fontSize: 13 },
  linkBold: { color: '#2a5298', fontWeight: '700' },
  backText: { textAlign: 'center', color: 'rgba(255,255,255,0.7)', marginTop: 25, fontSize: 14 },
});
