import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import ChatScreen from './src/ui/screens/ChatScreen';
import HistoryScreen from './src/ui/screens/HistoryScreen';
import ProfileScreen from './src/ui/screens/ProfileScreen';
import SettingsScreen from './src/ui/screens/SettingScreen';
import { ItemProvider } from './src/ui/context/ItemContext';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import ItemLocationScreen from './src/ui/screens/ItemLocationScreen';

export type RootStackParamList = {
  Chat: undefined;
  History: undefined;
  Profile: undefined;
  Settings: undefined;
  ItemLocation: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <SafeAreaProvider>
      <ItemProvider>
        <NavigationContainer>
          <Stack.Navigator
            id="RootStack"
            initialRouteName="Chat"
            screenOptions={{ headerShown: false }}
          >
            <Stack.Screen name="Chat" component={ChatScreen} />
            <Stack.Screen name="History" component={HistoryScreen} />
            <Stack.Screen name="Profile" component={ProfileScreen} />
            <Stack.Screen name="Settings" component={SettingsScreen} />
            <Stack.Screen name="ItemLocation" component={ItemLocationScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </ItemProvider>
    </SafeAreaProvider>
  );
}