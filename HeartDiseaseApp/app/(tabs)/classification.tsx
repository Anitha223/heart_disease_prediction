import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator, TouchableOpacity, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { API_BASE } from '../config';

export default function ClassificationScreen() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState({
        svm_acc: 0,
        dt_ac: 0,
        ann_ac: 0,
        hmm_ac: 0,
        best_model: ''
    });

    useEffect(() => {
        fetchClassificationData();
    }, []);

    const fetchClassificationData = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/classification`, {
                headers: {
                    'Content-Type': 'application/json',
                    'bypass-tunnel-reminder': 'true'
                }
            });
            const result = await response.json();
            if (result.success) {
                setData(result);
            } else {
                Alert.alert('Error', result.error || 'Failed to fetch classification details');
            }
        } catch (error) {
            Alert.alert('Connection Error', 'Ensure backend is running.');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <View style={styles.loader}>
                <ActivityIndicator size="large" color="#FF6B6B" />
                <Text style={{ marginTop: 10, color: '#2C3E50', fontWeight: 'bold' }}>Evaluating Models...</Text>
            </View>
        );
    }

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}>
            <Text style={styles.title}>Classification View</Text>
            <Text style={styles.subtitle}>Your safe and accurate identification</Text>

            <View style={styles.bestModelCard}>
                <Text style={styles.bestModelLabel}>Best Model:</Text>
                <Text style={styles.bestModelValue}>{data.best_model}</Text>
            </View>

            <View style={styles.tableCard}>
                <Text style={styles.tableTitle}>Model Accuracies</Text>
                
                <View style={styles.table}>
                    <View style={[styles.row, styles.headerRow]}>
                        <Text style={[styles.cell, styles.headerCell, { flex: 2.5 }]}>Model</Text>
                        <Text style={[styles.cell, styles.headerCell, { flex: 1, textAlign: 'center' }]}>Accuracy</Text>
                    </View>

                    {[
                        { name: 'Support Vector Machine (SVM)', acc: data.svm_acc },
                        { name: 'Decision Tree Classifier (DTC)', acc: data.dt_ac },
                        { name: 'Artificial Neural Network (ANN)', acc: data.ann_ac },
                        { name: 'Hidden Markov Model (HMM)', acc: data.hmm_ac }
                    ].map((item, index) => (
                        <View key={index} style={[styles.row, index % 2 === 0 ? styles.evenRow : {}]}>
                            <Text style={[styles.cell, { flex: 2.5, fontWeight: '500', color: '#34495E' }]}>{item.name}</Text>
                            <Text style={[styles.cell, styles.accValue, { flex: 1, textAlign: 'center' }]}>{item.acc}%</Text>
                        </View>
                    ))}
                </View>
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
    loader: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    
    title: { fontSize: 26, fontWeight: '800', color: '#1e3c72', textAlign: 'center' },
    subtitle: { fontSize: 13, color: '#888', textAlign: 'center', marginBottom: 25 },
    
    bestModelCard: {
        backgroundColor: '#FF6B6B', borderRadius: 16, padding: 20, marginBottom: 25,
        alignItems: 'center',
        shadowColor: '#FF6B6B', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.3, shadowRadius: 15, elevation: 6,
    },
    bestModelLabel: { fontSize: 13, color: 'rgba(255,255,255,0.85)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1 },
    bestModelValue: { fontSize: 22, fontWeight: '900', color: '#fff', marginTop: 5 },

    tableCard: {
        backgroundColor: '#fff', borderRadius: 18, padding: 20, marginBottom: 25,
        shadowColor: '#000', shadowOffset: { width: 0, height: 5 }, shadowOpacity: 0.08, shadowRadius: 10, elevation: 5,
    },
    tableTitle: { fontSize: 18, fontWeight: '700', color: '#2C3E50', marginBottom: 15 },
    
    table: {
        borderWidth: 1, borderColor: '#eee', borderRadius: 10, overflow: 'hidden'
    },
    row: {
        flexDirection: 'row', padding: 15, borderBottomWidth: 1, borderBottomColor: '#eee', alignItems: 'center'
    },
    evenRow: {
        backgroundColor: '#fafbfc'
    },
    headerRow: {
        backgroundColor: '#1e3c72', borderBottomWidth: 0
    },
    cell: {
        fontSize: 13,
    },
    headerCell: {
        color: '#fff', fontWeight: 'bold'
    },
    accValue: {
        fontWeight: 'bold', color: '#27ae60'
    },

    backBtn: { alignItems: 'center', marginTop: 10 },
    backText: { color: '#2a5298', fontSize: 15, fontWeight: '700' },
});
