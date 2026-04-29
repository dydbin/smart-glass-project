import React from 'react';
import { Modal, View, Text, StyleSheet, Pressable } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../../../App';

import { Image } from 'react-native';
import userIcon from '../icon/user.png';
import historyIcon from '../icon/history.png';
import settingsIcon from '../icon/setting.png';

type SidebarProps = {
  visible: boolean;
  onClose: () => void;
};

export default function Sidebar({ visible, onClose }: SidebarProps) {
  const navigation =
    useNavigation<NativeStackNavigationProp<RootStackParamList>>();

  const handleMove = (screen: keyof RootStackParamList) => {
    onClose();
    navigation.navigate(screen);
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View style={styles.sidebarWrapper}>
          <View style={styles.sidebar}>
            <Text style={styles.title}>메뉴</Text>
            <Pressable
              style={styles.menuItem}
              onPress={() => handleMove('History')}
            >
              <Text style={styles.menuText}>히스토리</Text>
              <Image source={historyIcon} style={styles.menuIconImage} />
            </Pressable>

            <Pressable
              style={styles.menuItem}
              onPress={() => handleMove('Profile')}
            >
              <Text style={styles.menuText}>사용자 정보</Text>
              <Image source={userIcon} style={styles.menuIconImage} />
            </Pressable>

            <Pressable
              style={styles.menuItem}
              onPress={() => handleMove('Settings')}
            >
              <Text style={styles.menuText}>설정</Text>
              <Image source={settingsIcon} style={styles.menuIconImage} />
            </Pressable>

            <View style={styles.logoutArea}>
              <Pressable style={styles.logoutButton}>
                <Text style={styles.logoutText}>↪ 로그아웃</Text>
              </Pressable>
            </View>
          </View>
        </View>
        <Pressable style={styles.backdrop} onPress={onClose} />
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'rgba(0,0,0,0.28)',
  },
  sidebarWrapper: {
    width: 280,
    height: '100%',
    backgroundColor: 'transparent',
  },
  sidebar: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    paddingTop: 72,
    paddingHorizontal: 20,
    borderTopRightRadius: 28,
    borderBottomRightRadius: 28,
  },
  backdrop: {
    flex: 1,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: '#111827',
    marginBottom: 28,
  },

  menuText: {
    fontSize: 16,
    color: '#111827',
    fontWeight: '500',
  },
  menuItem: {
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  menuIconImage: {
    width: 20,
    height: 20,
    resizeMode: 'contain',
  },

  logoutArea: {
    marginTop: 'auto',
    paddingBottom: 24,
  },

  logoutButton: {
    paddingVertical: 12,
  },

  logoutText: {
    fontSize: 14,
    color: '#333333',
    fontWeight: '600',
  },
});