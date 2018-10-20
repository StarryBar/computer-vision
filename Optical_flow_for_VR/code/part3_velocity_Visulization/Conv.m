% c=0: Convolution, c=1: Correlation
%can only deal with odd*odd filter
function [Iout] = Conv(Iin, Filter, c)
    if c == 0
        Filter = rot90(Filter,2);
    end
        
    [ri,ci] = size(Iin);
    [rf,cf] = size(Filter);
    Iout = zeros(ri,ci);
    for i=ceil(rf/2):1:ri-floor(rf/2)
       for j=ceil(cf/2):1:ci-floor(cf/2)
          tmp = sum(Iin(i-floor(rf/2):i+floor(rf/2), j-floor(cf/2):j+floor(cf/2)).*Filter);
          Iout(i,j)=sum(tmp);
          
       end
    end   
 end
