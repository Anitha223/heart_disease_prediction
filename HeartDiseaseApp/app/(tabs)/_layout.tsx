import { Tabs } from 'expo-router';
import React from 'react';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: { display: 'none' }, // Hide tab bar - we use custom navigation
      }}>
      <Tabs.Screen name="index" options={{ title: 'Welcome' }} />
      <Tabs.Screen name="login" options={{ title: 'Login' }} />
      <Tabs.Screen name="register" options={{ title: 'Register' }} />
      <Tabs.Screen name="home" options={{ title: 'Home' }} />
      <Tabs.Screen name="predict" options={{ title: 'Predict' }} />
      <Tabs.Screen name="about" options={{ title: 'About' }} />
      <Tabs.Screen name="classification" options={{ title: 'Classification View' }} />
      <Tabs.Screen name="explore" options={{ href: null }} />
    </Tabs>
  );
}
