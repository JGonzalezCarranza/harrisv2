#include "defines.hpp"

void QuicksortInverse(int *pOffsets, const int *pValues, int nLow, int nHigh)
{
        int i = nLow;
        int j = nHigh;

        float x = pValues[(int)pOffsets[(int)(nLow + nHigh) /2 ]];

        while (i <= j)
        {
                while (pValues[(int)pOffsets[i]] > x) i++;
                while (pValues[(int)pOffsets[j]] < x) j--;

                if (i <= j)
                {
                        const float temp = pOffsets[i];
                        pOffsets[i] = pOffsets[j];
                        pOffsets[j] = temp;

                        i++;
                        j--;
                }
        }

        if (nLow < j) QuicksortInverse(pOffsets, pValues, nLow, j);
        if (i < nHigh) QuicksortInverse(pOffsets, pValues, i, nHigh);
}
