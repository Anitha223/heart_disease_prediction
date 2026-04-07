import React from 'react';
import { StyleSheet, View, SafeAreaView, StatusBar, ActivityIndicator, Text, TouchableOpacity } from 'react-native';
import { WebView } from 'react-native-webview';
import { API_BASE } from '../config';

export default function App() {
  const [loading, setLoading] = React.useState(true);
  const [errorVisible, setErrorVisible] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState('');
  const webViewRef = React.useRef(null);

  const reloadWebView = () => {
    setErrorVisible(false);
    setLoading(true);
    webViewRef.current?.reload();
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      <WebView 
        ref={webViewRef}
        source={{ uri: API_BASE }} 
        style={styles.webview}
        onLoadStart={() => setLoading(true)}
        onLoadEnd={() => setLoading(false)}
        onError={(syntheticEvent) => {
          const { nativeEvent } = syntheticEvent;
          setErrorMessage(nativeEvent.description);
          setErrorVisible(true);
          setLoading(false);
        }}
        headers={{ 'bypass-tunnel-reminder': 'true' }}
        startInLoadingState={true}
      />

      {loading && (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#FF6B6B" />
          <Text style={{marginTop: 10, color: '#666'}}>Connecting to Server...</Text>
        </View>
      )}

      {errorVisible && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorIcon}>⚠️</Text>
          <Text style={styles.errorTitle}>Connection Failed</Text>
          <Text style={styles.errorText}>Unable to reach: {API_BASE}</Text>
          <Text style={styles.errorDetail}>{errorMessage}</Text>
          <TouchableOpacity style={styles.retryBtn} onPress={reloadWebView}>
            <Text style={styles.retryText}>RETRY CONNECTION</Text>
          </TouchableOpacity>
          <Text style={styles.tipText}>Tip: Ensure Laptop Hotspot is ON and Django is running.</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  webview: {
    flex: 1,
    backgroundColor: '#fff',
  },
  loader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  errorContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 25,
  },
  errorIcon: { fontSize: 50, marginBottom: 15 },
  errorTitle: { fontSize: 22, fontWeight: 'bold', color: '#333', marginBottom: 10 },
  errorText: { fontSize: 16, color: '#666', textAlign: 'center', marginBottom: 5 },
  errorDetail: { fontSize: 12, color: '#999', textAlign: 'center', marginBottom: 20 },
  retryBtn: {
    backgroundColor: '#FF6B6B',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    elevation: 3,
  },
  retryText: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
  tipText: { marginTop: 20, fontSize: 13, color: '#888', fontStyle: 'italic', textAlign: 'center' }
});
