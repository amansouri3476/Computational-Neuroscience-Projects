function output = lookupRC(v)


    switch str2double(sprintf('%d%d',v(1),v(2)))
        
        case 17
            output = 'A';
        case 18
            output = 'G';
        case 19
            output = 'M';
        case 110
            output = 'S';
        case 111
            output = 'Y';
        case 112
            output = '4';
        case 27
            output = 'B';
        case 28
            output = 'H';
        case 29
            output = 'N';
        case 210
            output = 'T';
        case 211
            output = 'Z';
        case 212
            output = '5';
        case 37
            output = 'C';
        case 38
            output = 'I';
        case 39
            output = 'O';
        case 310
            output = 'U';
        case 311
            output = '0';
        case 312
            output = '6';
        case 47
            output = 'D';
        case 48
            output = 'J';
        case 49
            output = 'P';
        case 410
            output = 'V';
        case 411
            output = '1';
        case 412
            output = '7';
        case 57
            output = 'E';
        case 58
            output = 'K';
        case 59
            output = 'Q';
        case 510
            output = 'W';
        case 511
            output = '2';
        case 512
            output = '8';
        case 67
            output = 'F';
        case 68
            output = 'L';
        case 69
            output = 'R';
        case 610
            output = 'X';
        case 611
            output = '3';
        case 612
            output = '9';

    end

end