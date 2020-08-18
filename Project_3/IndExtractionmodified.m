function output = IndExtractionmodified(s)

    output = find(diff(s.test(10,:)) > 0) + 1;

end