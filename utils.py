import datetime
import pandas as pd

from typing import Dict


TF_EQUIV = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "1h": "1H", "4h": "4H", "12h": "12H", "1d": "D"}

DAYS_TO_TF = {
    "1m": {1: 1440, 2: 2880, 3: 4320, 4: 5760, 5: 7200},
    "5m": {1: 288, 2: 576, 3: 864, 4: 1152, 5: 1440},
    "15m": {1: 96, 2: 192, 3: 288, 4: 384, 5: 480},
    "30m": {1: 48, 2: 96, 3: 144, 4: 192, 5: 240},
    "1h": {1: 24, 2: 48, 3: 72, 4: 96, 5: 120},
    "4h": {1: 6, 2: 12, 3: 18, 4: 24, 5: 30},
    "12h": {1: 2, 2: 4, 3: 6, 4: 8, 5: 10},
    "1d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
}

COIN_CATEGORIES = {
    "binance": {
        "defi": ['LUNAUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'RUNEUSDT', 'CRVUSDT', 'LRCUSDT', 'CELOUSDT', 'YFIUSDT', 'COMPUSDT', '1INCHUSDT', 'SUSHIUSDT', 'KAVAUSDT', 'SNXUSDT', 'ZRXUSDT', 'KNCUSDT', 'RENUSDT', 'SRMUSDT', 'RSRUSDT', 'BTCSTUSDT', 'ALPHAUSDT',
                 'REEFUSDT', 'BANDUSDT', 'BAKEUSDT', 'BALUSDT', 'YFIIUSDT', 'LINAUSDT', 'BTSUSDT', 'DODOUSDT', 'LITUSDT', 'TRBUSDT', 'AKROUSDT', 'BELUSDT', 'FLMUSDT', 'UNFIUSDT'],
        "gaming": ['MANAUSDT', 'AXSUSDT', 'SANDUSDT',
                   'GALAUSDT', 'ENJUSDT', 'ALICEUSDT', 'TLMUSDT'],
        "metaverse": ['MANAUSDT', 'AXSUSDT', 'SANDUSDT', 'ALICEUSDT'],
        "L1/L2": ['ETHUSDT', 'ETHUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'LUNAUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT',
                  'ALGOUSDT', 'TRXUSDT', 'ICPUSDT', 'VETUSDT', 'KLAYUSDT', 'XTZUSDT', 'ONEUSDT', 'EOSUSDT', 'KSMUSDT', 'NEOUSDT', 'CELOUSDT', 'WAVESUSDT', 'OMGUSDT', 'KAVAUSDT', 'ONTUSDT', 'SKLUSDT'],
        "nft": ['MANAUSDT', 'AXSUSDT', 'SANDUSDT', 'GALAUSDT', 'ENJUSDT',
                'CHZUSDT', 'CHRUSDT', 'ALICEUSDT', 'OGNUSDT', 'BAKEUSDT'],
        "polka": ['DOTUSDT', 'KSMUSDT', 'REEFUSDT', 'LITUSDT', 'AKROUSDT'],
        "bsc": ['BNBUSDT', 'GALAUSDT', 'ONTUSDT', 'SXPUSDT', 'BTCSTUSDT', 'BAKEUSDT', 'SFPUSDT', 'CTKUSDT', 'LINAUSDT', 'DODOUSDT', 'BELUSDT', 'UNFIUSDT'],
    },
    "kucoin": {
        "meta": ['SANDUSDTM', 'MANAUSDTM', 'ENJUSDTM', 'AXSUSDTM'],
        "defi": ['LINKUSDTM', 'CRVUSDTM', 'UNIUSDTM', 'LUNAUSDTM', 'MIRUSDTM', 'SUSHIUSDTM',
                 'AAVEUSDTM', 'YFIUSDTM', 'SNXUSDTM', 'MKRUSDTM', 'COMPUSDTM', 'WAVESUSDTM'],
        "polk": ['KSMUSDTM'],
        "nft": ['SANDUSDTM', 'MANAUSDTM', 'RNDRUSDTM', 'ENJUSDTM', 'THETAUSDTM', 'CHRUSDTM', 'CHZUSDTM'],
    },
}


def ms_to_dt(ms: int) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(ms / 1000)


def resample_timeframe(data: pd.DataFrame, tf: str) -> pd.DataFrame:
    return data.resample(TF_EQUIV[tf]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "bidPrice": "last", "askPrice": "last"}
    )


def z_score_idx(data: pd.DataFrame) -> Dict[str, int]:
    '''

    REDO THIS AND USE ABS FOR THE -1 and 1 to get the closest value! probably by
    some sort of abs(v-1) closer to 0 than previous value...

    '''
    z_score = {"idx0": 0, "idx1": 0, "idx-1": 0, "1": 2, "0": 1, "-1": -2}

    for i, v in enumerate(data):

        if v >= 0 and v < z_score["0"]:
            z_score['idx0'] = i
            z_score["0"] = v

        if v <= -1 and v > z_score["-1"]:
            z_score['idx-1'] = i
            z_score["-1"] = v

        if v >= 1 and v < z_score["1"]:
            z_score['idx1'] = i
            z_score["1"] = v

    if z_score["1"] == 2:
        z1 = min(data, key=lambda x: abs(1-x))
        z_score["idx1"] = data[data == z1].index[0]
        z_score["1"] = z1

    if z_score["0"] == 1:
        z0 = min(data, key=lambda x: abs(0-x))
        z_score["idx0"] = data[data == z0].index[0]
        z_score["0"] = z0

    if z_score["-1"] == -2:
        zm1 = min(data, key=lambda x: -1+x)
        z_score["idx-1"] = data[data == zm1].index[0]
        z_score["-1"] = zm1

    return z_score
