randmax=32767
random_next=1

function crand()
    global random_next
    global randmax
    #println("random_next=", random_next)
    random_next = div(random_next * 1103515245 + 12345, 65536) % randmax;
    return random_next
end

function crand01()
    return crand() / randmax
end

function crandn()
    crand() / randmax*5.2-2.6
end

function printrand()
    for i in 1:20
        println(crandn())
    end
end
