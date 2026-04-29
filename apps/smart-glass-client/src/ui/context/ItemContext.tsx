import React, { createContext, useContext, useState } from 'react';

type ItemData = {
  count: number;
  lastTime: number;
};

type ItemContextType = {
  itemCounts: Record<string, ItemData>;
  addItem: (item: string) => void;
};

const ItemContext = createContext<ItemContextType | null>(null);

export const ItemProvider = ({ children }: { children: React.ReactNode }) => {
  const [itemCounts, setItemCounts] = useState<Record<string, ItemData>>({});

  const addItem = (item: string) => {
    setItemCounts((prev) => {
      const existing = prev[item];

      return {
        ...prev,
        [item]: {
          count: existing ? existing.count + 1 : 1,
          lastTime: Date.now(),
        },
      };
    });
  };

  return (
    <ItemContext.Provider value={{ itemCounts, addItem }}>
      {children}
    </ItemContext.Provider>
  );
};

export const useItemContext = () => {
  const context = useContext(ItemContext);

  if (!context) {
    throw new Error('ItemContext 사용 오류');
  }

  return context;
};