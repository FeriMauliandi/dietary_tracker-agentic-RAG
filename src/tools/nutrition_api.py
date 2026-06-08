import requests
import re
from typing import List, Dict, Any
from src.core.config import settings

# --- HELPER 1: PENCARIAN TOKEN DI FATSECRET ---
def get_fatsecret_token() -> str:
    url = "https://oauth.fatsecret.com/connect/token"
    data = {"grant_type": "client_credentials"}
    auth = (settings.FATSECRET_CLIENT_ID, settings.FATSECRET_CLIENT_SECRET)
    
    try:
        response = requests.post(url, data=data, auth=auth)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"⚠️ [Auth] Gagal mendapatkan token FatSecret: {e}")
        return None

# --- HELPER 2: PENCARIAN 1 ITEM DI FATSECRET ---
def get_fatsecret_item(item_name: str, token: str) -> Dict[str, Any]:
    if not token:
        return {"found": False}
        
    url = "https://platform.fatsecret.com/rest/server.api"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "method": "foods.search",
        "search_expression": item_name,
        "format": "json",
        "max_results": 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        foods = response.json().get("foods", {}).get("food", [])
        
        if foods:
            food = foods[0] if isinstance(foods, list) else foods
            desc = food.get("food_description", "")
            
            cal = float((re.search(r"Calories:\s*([\d.]+)kcal", desc) or [0, 0])[1])
            pro = float((re.search(r"Protein:\s*([\d.]+)g", desc) or [0, 0])[1])
            car = float((re.search(r"Carbs:\s*([\d.]+)g", desc) or [0, 0])[1])
            
            return {"found": True, "cal": cal, "pro": pro, "car": car, "source": "FatSecret"}
    except Exception:
        pass
    return {"found": False}

# --- HELPER 3: PENCARIAN 1 ITEM DI USDA ---
def get_usda_item(item_name: str) -> Dict[str, Any]:
    api_key = settings.USDA_API_KEY
    if not api_key:
        return {"found": False}

    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key": api_key, "query": item_name, "pageSize": 1}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "foods" in data and len(data["foods"]) > 0:
            nutrients = data["foods"][0].get("foodNutrients", [])
            
            cal = next((n['value'] for n in nutrients if n['nutrientName'] == 'Energy' and n['unitName'] == 'KCAL'), 0)
            pro = next((n['value'] for n in nutrients if n['nutrientName'] == 'Protein'), 0)
            car = next((n['value'] for n in nutrients if n['nutrientName'] == 'Carbohydrate, by difference'), 0)
            
            return {"found": True, "cal": cal, "pro": pro, "car": car, "source": "USDA"}
    except Exception:
        pass
    return {"found": False}

# --- FUNGSI UTAMA (GABUNGAN) ---
def fetch_combined_nutrition_data(items_data: List[Dict[str, str]]) -> Dict[str, Any]:
    print(f"[Tool] Memulai pencarian gabungan (FatSecret + USDA Fallback)...")
    token = get_fatsecret_token()
    
    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    found_items_log = []

    # Looping menggunakan list of dictionaries
    for item_dict in items_data:
        # Pengecekan tipe data jaga-jaga jika ada input kotor
        if isinstance(item_dict, str):
            nama_asli = item_dict
            nama_inggris = item_dict
        else:
            nama_asli = item_dict.get("asli", "")
            nama_inggris = item_dict.get("english", "")
            
        # 1. Coba FatSecret dulu (Bahasa Indonesia)
        result = get_fatsecret_item(nama_asli, token)
        
        # 2. Jika gagal di FatSecret, Fallback ke USDA (Bahasa Inggris)
        if not result["found"]:
            print(f"🔄 '{nama_asli}' tidak ditemukan di FatSecret. Beralih ke USDA dengan kueri '{nama_inggris}'...")
            result = get_usda_item(nama_inggris)
            
        # 3. Kalkulasi hasil akhir
        if result["found"]:
            total_calories += result["cal"]
            total_protein += result["pro"]
            total_carbs += result["car"]
            found_items_log.append(f"{nama_asli} ({result['cal']} kkal dari {result['source']})")
        else:
            found_items_log.append(f"{nama_asli} (Tidak ditemukan)")

    summary_text = (
        f"Data diekstraksi untuk {len(items_data)} item. "
        f"Status: {', '.join(found_items_log)}. "
        f"Estimasi Total: {total_calories:.1f} Kalori, {total_protein:.1f}g Protein, {total_carbs:.1f}g Karbohidrat."
    )
    
    return {"summary": summary_text}